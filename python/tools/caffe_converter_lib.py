from lib.proto import mace_pb2
from lib.proto import caffe_pb2
from lib.python.tools import memory_optimizer
import google.protobuf.text_format
import numpy as np
import math

# TODO: support NCHW formt, now only support NHWC.
padding_mode = {
  'VALID': 0,
  'SAME': 1,
  'FULL': 2
}
pooling_type_mode = {
  'AvgPool': 1,
  'MaxPool': 2
}

buffer_type_map = {
  'CONV2D_FILTER' : 0,
  'IN_OUT_CHANNEL' : 1,
  'ARGUMENT' : 2,
  'IN_OUT_HEIGHT' : 3,
  'IN_OUT_WIDTH' : 4,
  'WINOGRAD_FILTER' : 5,
  'DW_CONV2D_FILTER' : 6,
  'WEIGHT_HEIGHT' : 7,
}

data_type_map = {
  'DT_HALF' : mace_pb2.DT_HALF,
  'DT_FLOAT': mace_pb2.DT_FLOAT
}

activation_name_map = {
  'ReLU' : 'RELU',
  'PReLU' : 'PRELU',
  'Sigmoid' : 'SIGMOID',
  'TanH' : 'TANH',
}

MACE_INPUT_NODE_NAME = "mace_input_node"
MACE_OUTPUT_NODE_NAME = "mace_output_node"

OPENCL_IMAGE_MAX_SIZE = 16384

class Operator(object):
  def __init__(self, name, type, layer):
    self.name = name
    self.type = type
    self.layer = layer
    self.parents = []
    self.children = []
    self.data = []

  def add_parent(self, parent_op):
    assert parent_op not in self.parents
    self.parents.append(parent_op)
    if self not in parent_op.children:
      parent_op.children.append(self)

  def add_child(self, child_op):
    assert child_op not in self.children
    self.children.append(child_op)
    if self not in child_op.parents:
      child_op.parents.append(self)

def BlobToNPArray(blob):
  if blob.num != 0:
    return (np.asarray(blob.data, dtype=np.float32).
            reshape(blob.num, blob.channels, blob.height, blob.width))
  else:
    return np.asarray(blob.data, dtype=np.float32).reshape(blob.shape.dim)

def CommonConvert(op, mace_type, dt):
  op_def = mace_pb2.OperatorDef()
  arg = op_def.arg.add()
  arg.name = 'T'
  arg.i = dt
  data_format_arg = op_def.arg.add()
  data_format_arg.name = 'data_format'
  data_format_arg.s = 'NHWC'
  op_def.name = op.name
  op_def.type = mace_type
  op_def.input.extend([parent.name+':0' for parent in op.parents])
  return op_def

class CaffeConverter(object):
  def __init__(self, caffe_net, weights, net_def, dt, device, winograd):
    self.net_def = net_def
    self.caffe_net = caffe_net
    self.weights = weights
    self.dt = dt
    self.device = device
    self.winograd = winograd
    self.resolved_ops = set()

    layers = caffe_net.layer

    # remove train layers and dropout
    layers = self.remove_unused_layers(layers)

    # Construct graph
    # Only support single-output layer
    # layer with single output often use the same top name.
    self.ops = [Operator(layer.name, layer.type, layer) for layer in layers]
    self.ops_map = {op.name : op for op in self.ops}
    output_op = {}
    for layer in layers:
      op = self.ops_map[layer.name]
      for input_name in layer.bottom:
        assert input_name != layer.name
        parent_op = output_op.get(input_name)
        if parent_op is None:
          parent_op = self.ops_map[input_name]
        op.add_parent(parent_op)
      if len(layer.top) > 1:
        raise Exception('Only support single-output layers')
      for output_name in layer.top:
        if output_name == layer.name:
          continue
        output_op[output_name] = op

    # Load weights
    weights_layers = weights.layer
    for layer in weights_layers:
      if not layer.blobs:
        continue
      if layer.name in self.ops_map:
        op = self.ops_map[layer.name]
        op.data = [BlobToNPArray(blob) for blob in layer.blobs]

    # toposort ops
    self.ops = self.toposort_ops()

  def remove_unused_layers(self, layers):
    phase_map = {0: 'train', 1: 'test'}
    test_layers_names = set()
    test_layers = []
    for layer in layers:
      phase = 'test'
      if len(layer.include):
        phase = phase_map[layer.include[0].phase]
      if len(layer.exclude):
        phase = phase_map[layer.exclude[0].phase]
      if phase == 'test' and layer.type != 'Dropout':
        test_layers.append(layer)
        assert layer.name not in test_layers_names
        test_layers_names.add(layer.name)
    return test_layers

  def toposort_ops(self):
    sorted_ops = []
    temp_visited = set()
    visited = set()

    def search(op):
      if op.name in temp_visited:
        raise Exception("The model is not DAG")
      if op.name in visited:
        return
      temp_visited.add(op.name)
      for parent_op in op.parents:
        search(parent_op)
      temp_visited.remove(op.name)
      sorted_ops.append(op)
      visited.add(op.name)

    for op in self.ops:
      search(op)

    return sorted_ops


  def add_buffer_to_image(self, input_name, input_type):
    output_name = input_name[:-2] + "_b2i" + input_name[-2:]
    op_def = self.net_def.op.add()
    op_def.name = output_name[:-2]
    op_def.type = 'BufferToImage'
    op_def.input.extend([input_name])
    op_def.output.extend([output_name])

    arg = op_def.arg.add()
    arg.name = 'buffer_type'
    arg.i = buffer_type_map[input_type]
    arg = op_def.arg.add()
    arg.name = 'mode'
    arg.i = 0
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    return output_name

  def add_image_to_buffer(self, input_name, input_type):
    output_name = input_name[:-2] + "_i2b" + input_name[-2:]
    op_def = self.net_def.op.add()
    op_def.name = output_name[:-2]
    op_def.type = 'ImageToBuffer'
    op_def.input.extend([input_name])
    op_def.output.extend([output_name])

    arg = op_def.arg.add()
    arg.name = 'buffer_type'
    arg.i = buffer_type_map[input_type]
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    return output_name

  def add_input_transform(self, name):
    new_input_name = MACE_INPUT_NODE_NAME + ":0"
    op_def = self.net_def.op.add()
    op_def.name = name
    op_def.type = 'BufferToImage'
    op_def.input.extend([new_input_name])
    if name not in self.ops_map:
      raise Exception("Input name not in the model")
    top_name = self.ops_map[name].layer.top[0]
    op_def.output.extend([top_name+':0'])

    epsilon_arg = op_def.arg.add()
    epsilon_arg.name = 'buffer_type'
    epsilon_arg.i = buffer_type_map['IN_OUT_CHANNEL']

    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt

  def add_output_transform(self, name):
    output_name = MACE_OUTPUT_NODE_NAME + ":0"
    op_def = self.net_def.op.add()
    op_def.name = output_name[:-2]
    op_def.type = 'ImageToBuffer'
    op_def.input.extend([name+':0'])
    op_def.output.extend([output_name])

    epsilon_arg = op_def.arg.add()
    epsilon_arg.name = 'buffer_type'
    epsilon_arg.i = buffer_type_map['IN_OUT_CHANNEL']

  def add_tensor(self, name, value):
    tensor = self.net_def.tensors.add()
    tensor.name = name

    shape = list(value.shape)
    tensor.dims.extend(shape)

    tensor.data_type = mace_pb2.DT_FLOAT
    tensor.float_data.extend(value.flat)

  def add_stride_pad_kernel_arg(self, param, op_def):
    try:
      if len(param.stride) > 1 or len(param.kernel_size) > 1 or len(param.pad) > 1:
        raise Exception('Mace does not support multiple stride/kernel_size/pad')
      stride = param.stride[0] if len(param.stride) else 1
      pad = param.pad[0] if len(param.pad) else 0
      kernel = param.kernel_size[0] if len(param.kernel_size) else 0
    except TypeError:
      stride = param.stride
      pad = param.pad
      kernel = param.kernel_size

    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    if param.HasField("stride_h") or param.HasField("stride_w"):
      strides_arg.ints.extend([param.stride_h, param.stride_w])
    else:
      strides_arg.ints.extend([stride, stride])
    # Pad
    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding_values'
    if param.HasField("pad_h") or param.HasField("pad_w"):
      padding_arg.ints.extend([param.pad_h, param.pad_w])
    else:
      padding_arg.ints.extend([pad, pad])
    # kernel
    if op_def.type == 'Pooling':
      kernel_arg = op_def.arg.add()
      kernel_arg.name = 'kernels'
      if param.HasField("kernel_h") or param.HasField("kernel_w"):
        kernel_arg.ints.extend([param.kernel_h, param.kernel_w])
      else:
        kernel_arg.ints.extend([kernel, kernel])

  def convert_conv2d(self, op):
    op_def = CommonConvert(op, 'Conv2D', self.dt)
    param = op.layer.convolution_param

    # Add filter
    weight_tensor_name = op.name + '_weight:0'
    weight_data = op.data[0].transpose((2, 3, 0, 1))
    self.add_tensor(weight_tensor_name, weight_data)

    if self.device == 'gpu':
      buffer_type = "CONV2D_FILTER"
      output_name = self.add_buffer_to_image(weight_tensor_name, buffer_type)
      op_def.input.extend([output_name])
    else:
      op_def.input.extend([weight_tensor_name])

    # Add Bias
    if len(op.data) == 2:
      bias_tensor_name = op.name + '_bias:0'
      bias_data = op.data[1]
      self.add_tensor(bias_tensor_name, bias_data)
      if self.device == 'gpu':
        output_name = self.add_buffer_to_image(bias_tensor_name, "ARGUMENT")
        op_def.input.extend([output_name])
      else:
        op_def.input.extend([bias_tensor_name])

    self.add_stride_pad_kernel_arg(param, op_def)
    if len(param.dilation) > 0:
      dilation_arg = op_def.arg.add()
      dilation_arg.name = 'dilations'
      if len(param.dilation) == 1:
        dilation_arg.ints.extend([param.dilation[0], param.dilation[0]])
      elif len(param.dilation) == 2:
        dilation_arg.ints.extend([param.dilation[0], param.dilation[1]])
    final_op = op
    self.resolved_ops.add(op.name)

    if len(self.ops_map[final_op.name].children) == 1 \
        and self.ops_map[final_op.name].children[0].type in activation_name_map:
      activation_op = self.ops_map[final_op.name].children[0]
      op_def.type = "FusedConv2D"
      fused_act_arg = op_def.arg.add()
      fused_act_arg.name = 'activation'
      fused_act_arg.s = activation_name_map[activation_op.type]
      if activation_op.type == 'PReLU':
        alpha_arg = op_def.arg.add()
        alpha_arg.name = 'alpha'
        alpha_arg.f = activation_op.data[0][0]
      final_op = activation_op
      self.resolved_ops.add(activation_op.name)

    op_def.output.extend([final_op.name+':0'])
    self.net_def.op.extend([op_def])

  def convert_batchnorm(self, op):
    if len(op.children) != 1 or op.children[0].type != 'Scale':
      raise Exception('Now only support BatchNorm+Scale')
    op_def = CommonConvert(op, 'FoldedBatchNorm', self.dt)
    scale_op = op.children[0]

    epsilon_value = op.layer.batch_norm_param.eps
    if op.data[2][0] != 0:
      mean_value = (1. / op.data[2][0]) * op.data[0]
      var_value = (1. / op.data[2][0]) * op.data[1]
    else:
      raise RuntimeError('scalar is zero.')

    gamma_value = scale_op.data[0]
    beta_value = np.zeros_like(mean_value)
    if len(scale_op.data) == 2:
      beta_value = scale_op.data[1]

    scale_value = (
      (1.0 / np.vectorize(math.sqrt)(var_value + epsilon_value)) *
      gamma_value)
    offset_value = (-mean_value * scale_value) + beta_value
    input_names = [op.name+'_scale:0', op.name+'_offset:0']
    self.add_tensor(input_names[0], scale_value)
    self.add_tensor(input_names[1], offset_value)

    if self.device == 'gpu':
      for name in input_names:
        output_name = self.add_buffer_to_image(name, "ARGUMENT")
        op_def.input.extend([output_name])
    else:
      op_def.input.extend([name for name in input_names])

    self.resolved_ops.add(op.name)
    self.resolved_ops.add(scale_op.name)
    final_op = scale_op

    if len(self.ops_map[final_op.name].children) == 1 \
        and self.ops_map[final_op.name].children[0].type in activation_name_map:
      activation_op = self.ops_map[final_op.name].children[0]
      fused_act_arg = op_def.arg.add()
      fused_act_arg.name = 'activation'
      fused_act_arg.s = activation_name_map[activation_op.type]
      if activation_op.type == 'PReLU':
        alpha_arg = op_def.arg.add()
        alpha_arg.name = 'alpha'
        alpha_arg.f = activation_op.data[0][0]
      final_op = activation_op
      self.resolved_ops.add(activation_op.name)

    op_def.output.extend([final_op.name + ':0'])
    self.net_def.op.extend([op_def])

  def convert_inner_product(self, op):
    param = op.layer.inner_product_param
    try:
      if param.axis != 1 or param.transpose:
        raise ValueError('Do not support non-default axis and transpose '
                         'case for innner product')
    except AttributeError:
      pass

    op_def = CommonConvert(op, 'FC', self.dt)
    weight_tensor_name = op.name + '_weight:0'
    if op.data[0].ndim not in [2, 4]:
      raise ValueError('Unexpected weigth ndim.')
    if op.data[0].ndim == 4 and list(op.data[0].shape[:2] != [1, 1]):
      raise ValueError('Do not support 4D weight with shape [1, 1, *, *]')
    weight_data = op.data[0].reshape(-1, op.data[0].shape[-1])
    self.add_tensor(weight_tensor_name, weight_data)
    if self.device == 'gpu':
      buffer_type = "WEIGHT_HEIGHT"
      output_name = self.add_buffer_to_image(weight_tensor_name, buffer_type)
      op_def.input.extend([output_name])
    else:
      op_def.input.extend([weight_tensor_name])

    # Add Bias
    if len(op.data) == 2:
      bias_tensor_name = op.name + '_bias:0'
      bias_data = op.data[1]
      self.add_tensor(bias_tensor_name, bias_data)
      if self.device == 'gpu':
        output_name = self.add_buffer_to_image(bias_tensor_name, "ARGUMENT")
        op_def.input.extend([output_name])
      else:
        op_def.input.extend([bias_tensor_name])

    self.resolved_ops.add(op.name)
    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])

  def convert_pooling(self, op):
    op_def = CommonConvert(op, 'Pooling', self.dt)

    param = op.layer.pooling_param
    self.add_stride_pad_kernel_arg(param, op_def)
    if param.pool == caffe_pb2.PoolingParameter.MAX:
      pooling_type = "MaxPool"
    elif param.pool == caffe_pb2.PoolingParameter.AVE:
      pooling_type = "AvgPool"
    pooling_type_arg = op_def.arg.add()
    pooling_type_arg.name = 'pooling_type'
    pooling_type_arg.i = pooling_type_mode[pooling_type]

    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_activation(self, op):
    op_def = CommonConvert(op, 'Activation', self.dt)
    activation_arg = op_def.arg.add()
    activation_arg.name = 'activation'
    activation_arg.s = activation_name_map[op.type]
    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_prelu(self, op):
    op_def = CommonConvert(op, 'Activation', self.dt)
    activation_arg = op_def.arg.add()
    activation_arg.name = 'activation'
    activation_arg.s = activation_name_map[op.type]
    max_limit_arg = op_def.arg.add()
    max_limit_arg.name = 'alpha'
    max_limit_arg.f = op.data[0][0]
    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_add(self, op):
    op_def = CommonConvert(op, 'AddN', self.dt)
    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_concat(self, op):
    op_def = CommonConvert(op, 'Concat', self.dt)
    axis_arg = op_def.arg.add()
    axis_arg.name = 'axis'
    axis_arg.i = 3
    try:
      if op.layer.concat_param.HasFeild('axis'):
        axis_arg.i = op.concat_param.axis
      elif op.layer.concat_param.HasFeild('concat_dim'):
        axis_arg.i = op.concat_param.concat_dim
    except AttributeError:
      pass

    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_eltwise(self, op):
    op_def = CommonConvert(op, 'Eltwise', self.dt)
    param = op.layer.eltwise_param
    type_arg = op_def.arg.add()
    type_arg.name = 'type'
    type_arg.i = param.operation
    if len(param.coeff) > 0:
      coeff_arg = op_def.arg.add()
      coeff_arg.name = 'coeff'
      coeff_arg.ints.extend(list(param.coeff))

    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_normal_op(self, op):
    op_def = CommonConvert(op, op.type, self.dt)
    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def replace_in_out_name(self, input_name, output_name):
    input_name = input_name + ":0"
    output_name = output_name + ":0"
    for op in self.net_def.op:
      if len(op.input) > 0 and op.input[0] == input_name:
        op.input[0] = MACE_INPUT_NODE_NAME + ":0"
      if len(op.output) > 0 and op.output[0] == output_name:
        op.output[0] = MACE_OUTPUT_NODE_NAME + ":0"

  def convert(self, input_node, output_node):
    if self.device == 'gpu':
      self.add_input_transform(input_node)

    assert self.ops[0].type == 'Input'

    for op in self.ops:
      if op.name in self.resolved_ops:
        continue
      if op.type == 'Input':
        self.resolved_ops.add(op.name)
      elif op.type == 'Convolution':
        self.convert_conv2d(op)
      elif op.type == 'BatchNorm':
        self.convert_batchnorm(op)
      elif op.type == 'InnerProduct':
        self.convert_inner_product(op)
      elif op.type == 'Pooling':
        self.convert_pooling(op)
      elif op.type == 'PReLU':
        self.convert_prelu(op)
      elif op.type in ['ReLU', 'Sigmoid', 'TanH']:
        self.convert_activation(op)
      elif op.type == 'Add':
        self.convert_add(op)
      elif op.type == 'Concat':
        self.convert_concat(op)
      elif op.type == 'Eltwise':
        self.convert_eltwise(op)
      elif op.type in ['Softmax']:
       self.convert_normal_op(op)
      else:
        raise Exception('Unknown Op: %s, type: %s' % (op.name, op.type))

    if self.device == 'gpu':
      self.add_output_transform(output_node)

    if self.device == 'cpu':
      self.replace_in_out_name(input_node, output_node)

    for op in self.ops:
      if op.name not in self.resolved_ops:
        print 'Unresolve Op: %s with type %s' % (op.name, op.type)


def convert_to_mace_pb(model_file, weight_file, input_node, output_node, data_type, device, winograd):
  net_def = mace_pb2.NetDef()
  dt = data_type_map[data_type]

  caffe_net = caffe_pb2.NetParameter()
  with open(model_file, "r") as f:
    google.protobuf.text_format.Merge(str(f.read()), caffe_net)

  weights = caffe_pb2.NetParameter()
  with open(weight_file, "rb") as f:
    weights.MergeFromString(f.read())

  converter = CaffeConverter(caffe_net, weights, net_def, dt, device, winograd)
  converter.convert(input_node, output_node)
  print "PB Converted."
  if device == 'gpu':
    print "start optimize memory."
    mem_optimizer = memory_optimizer.MemoryOptimizer(net_def)
    mem_optimizer.optimize()
    print "Memory optimization done."

  return net_def
