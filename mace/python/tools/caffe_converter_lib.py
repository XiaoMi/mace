from mace.proto import mace_pb2
from mace.proto import caffe_pb2
from mace.python.tools import memory_optimizer
import google.protobuf.text_format
import numpy as np
import math

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
    self.output_shape_map = {}

  def add_parent(self, parent_op):
    self.parents.append(parent_op)
    parent_op.children.append(self)

  def get_single_parent(self):
    if len(self.parents) != 1:
      raise Exception('Operation %s expected single parent, but got %s'
                      % (self.name, len(self.parents)))
    return self.parents[0]

def BlobToNPArray(blob):
  if blob.num != 0:
    return (np.asarray(blob.data, dtype=np.float32).
            reshape((blob.num, blob.channels, blob.height, blob.width)))
  else:
    return np.asarray(blob.data, dtype=np.float32).reshape(blob.shape.dim)


class Shapes(object):
  @staticmethod
  def conv_pool_shape(input_shape, filter_shape, paddings, strides, dilations, round_func):
    output_shape = np.zeros_like(input_shape)
    output_shape[0] = input_shape[0]
    output_shape[1] = int(round_func((input_shape[1] + paddings[0] - filter_shape[0]
                       - (filter_shape[0] - 1) * (dilations[0] - 1)) / float(strides[0]))) + 1
    output_shape[2] = int(round_func((input_shape[2] + paddings[1] - filter_shape[1]
                       - (filter_shape[1] - 1) * (dilations[1] - 1)) / float(strides[1]))) + 1
    output_shape[3] = filter_shape[2]
    return output_shape

  @staticmethod
  def fully_connected_shape(input_shape, weight_shape):
    return [input_shape[0], 1, 1, weight_shape[0]]

  @staticmethod
  def concat_shape(input_shapes, axis):
    output_shape = None
    for input_shape in input_shapes:
      if output_shape is None:
        output_shape = list(input_shape)
      else:
        output_shape[axis] += input_shape[axis]
    return output_shape

  @staticmethod
  def slice_shape(input_shape, num_output):
    return [input_shape[0], input_shape[1], input_shape[2], input_shape[3]/num_output]

# outputs' name is [op.name + '_' + #]
class CaffeConverter(object):
  def __init__(self, caffe_net, weights, net_def, dt, device, winograd):
    self.net_def = net_def
    self.caffe_net = caffe_net
    self.weights = weights
    self.dt = dt
    self.device = device
    self.winograd = winograd
    self.resolved_ops = set()
    self.ops = []
    self.inputs_map = {} # caffe op name -> mace inputs' name

    # Add Input operations
    top_name_map = {}
    inputs = caffe_net.input
    for input in inputs:
      self.ops.extend([Operator(input, 'Input', None)])
      top_name_map[input] = input

    layers = caffe_net.layer
    # remove train layers and dropout
    layers = self.remove_unused_layers(layers)

    # Construct graph
    # Only support single-output layer
    # layer with single output often use the same top name.
    self.ops.extend([Operator(layer.name, layer.type, layer) for layer in layers])

    self.ops_map = {op.name : op for op in self.ops}
    output_op_map = {}
    for layer in layers:
      op = self.ops_map[layer.name]
      for input_name in layer.bottom:
        assert input_name != layer.name
        parent_op = output_op_map.get(input_name)
        if parent_op is None:
          parent_op = self.ops_map[input_name]
        op.add_parent(parent_op)
        if op.name not in self.inputs_map:
          self.inputs_map[op.name] = []
        self.inputs_map[op.name].extend([top_name_map[input_name]])
      for i in range(len(layer.top)):
        output_name = layer.top[i]
        if len(layer.top) == 1:
          top_name_map[output_name] = op.name
        else:
          top_name_map[output_name] = op.name + '_' + str(i)
        if output_name == layer.name:
          continue
        output_op_map[output_name] = op


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

  def CommonConvert(self, op, mace_type):
    op_def = mace_pb2.OperatorDef()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
    op_def.name = op.name
    op_def.type = mace_type
    op_def.input.extend([name+':0' for name in self.inputs_map[op.name]])
    return op_def

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

  def add_input_transform(self, names, is_single):
    for name in names:
      if is_single:
        new_input_name = MACE_INPUT_NODE_NAME + ":0"
      else:
        new_input_name = MACE_INPUT_NODE_NAME + '_' + name + ":0"
      op_def = self.net_def.op.add()
      op_def.name = name
      op_def.type = 'BufferToImage'
      op_def.input.extend([new_input_name])
      op_def.output.extend([name+':0'])

      epsilon_arg = op_def.arg.add()
      epsilon_arg.name = 'buffer_type'
      epsilon_arg.i = buffer_type_map['IN_OUT_CHANNEL']

      arg = op_def.arg.add()
      arg.name = 'T'
      arg.i = self.dt

  def add_output_transform(self, names, is_single):
    for name in names:
      if is_single:
        output_name = MACE_OUTPUT_NODE_NAME + ":0"
      else:
        output_name = MACE_OUTPUT_NODE_NAME + '_' + name + ":0"
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

  @staticmethod
  def add_output_shape(op_def, output_shape):
    mace_output_shape = mace_pb2.OutputShape()
    mace_output_shape.dims.extend(output_shape)
    op_def.output_shape.extend([mace_output_shape])

  def add_stride_pad_kernel_arg(self, param, op_def):
    try:
      if len(param.stride) > 1 or len(param.kernel_size) > 1 or len(param.pad) > 1:
        raise Exception('Mace does not support multiple stride/kernel_size/pad')
      stride = [param.stride[0], param.stride[0]] if len(param.stride) else [1, 1]
      pad = [param.pad[0] * 2, param.pad[0] * 2] if len(param.pad) else [0, 0]
      kernel = [param.kernel_size[0], param.kernel_size[0]] if len(param.kernel_size) else [0, 0]
    except TypeError:
      stride = [param.stride, param.stride]
      pad = [param.pad * 2, param.pad * 2]
      kernel = [param.kernel_size, param.kernel_size]

    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    if param.HasField("stride_h") or param.HasField("stride_w"):
      stride = [param.stride_h, param.stride_w]
    strides_arg.ints.extend(stride)
    # Pad
    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding_values'
    if param.HasField("pad_h") or param.HasField("pad_w"):
      pad = [param.pad_h * 2, param.pad_w * 2]
    padding_arg.ints.extend(pad)
    # kernel
    if op_def.type == 'Pooling':
      kernel_arg = op_def.arg.add()
      kernel_arg.name = 'kernels'
      if param.HasField("kernel_h") or param.HasField("kernel_w"):
        kernel = [param.kernel_h, param.kernel_w]
      kernel_arg.ints.extend(kernel)
    return pad, stride, kernel

  def convert_conv2d(self, op):
    op_def = self.CommonConvert(op, 'Conv2D')
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
      bias_data = op.data[1].reshape(-1)
      self.add_tensor(bias_tensor_name, bias_data)
      if self.device == 'gpu':
        output_name = self.add_buffer_to_image(bias_tensor_name, "ARGUMENT")
        op_def.input.extend([output_name])
      else:
        op_def.input.extend([bias_tensor_name])

    paddings, strides, _ = self.add_stride_pad_kernel_arg(param, op_def)
    dilations = [1, 1]
    if len(param.dilation) > 0:
      dilation_arg = op_def.arg.add()
      dilation_arg.name = 'dilations'
      if len(param.dilation) == 1:
        dilations = [param.dilation[0], param.dilation[0]]
      elif len(param.dilation) == 2:
        dilations = [param.dilation[0], param.dilation[1]]
      dilation_arg.ints.extend(dilations)
    final_op = op
    self.resolved_ops.add(op.name)

    output_shape = Shapes.conv_pool_shape(op.get_single_parent().output_shape_map[op.layer.bottom[0]],
                                          weight_data.shape,
                                          paddings, strides, dilations,
                                          math.floor)
    op.output_shape_map[op.layer.top[0]] = output_shape

    if len(self.ops_map[final_op.name].children) == 1 \
        and self.ops_map[final_op.name].children[0].type in activation_name_map:
      activation_op = self.ops_map[final_op.name].children[0]
      op_def.type = "FusedConv2D"
      fused_act_arg = op_def.arg.add()
      fused_act_arg.name = 'activation'
      fused_act_arg.s = activation_name_map[activation_op.type]
      final_op = activation_op
      final_op.output_shape_map[final_op.layer.top[0]] = output_shape
      self.resolved_ops.add(activation_op.name)

    op_def.output.extend([final_op.name+':0'])
    self.add_output_shape(op_def, output_shape)
    self.net_def.op.extend([op_def])

  def convert_batchnorm(self, op):
    if len(op.children) != 1 or op.children[0].type != 'Scale':
      raise Exception('Now only support BatchNorm+Scale')
    op_def = self.CommonConvert(op, 'FoldedBatchNorm')
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
      gamma_value).reshape(-1)
    offset_value = ((-mean_value * scale_value) + beta_value).reshape(-1)
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

    output_shape = op.get_single_parent().output_shape_map[op.layer.bottom[0]]

    if len(self.ops_map[final_op.name].children) == 1 \
        and self.ops_map[final_op.name].children[0].type in activation_name_map:
      activation_op = self.ops_map[final_op.name].children[0]
      fused_act_arg = op_def.arg.add()
      fused_act_arg.name = 'activation'
      fused_act_arg.s = activation_name_map[activation_op.type]
      final_op = activation_op
      final_op.output_shape_map[final_op.layer.top[0]] = output_shape
      self.resolved_ops.add(activation_op.name)

    op_def.output.extend([final_op.name + ':0'])
    self.add_output_shape(op_def, output_shape)
    self.net_def.op.extend([op_def])

  def convert_inner_product(self, op):
    param = op.layer.inner_product_param
    try:
      if param.axis != 1 or param.transpose:
        raise ValueError('Do not support non-default axis and transpose '
                         'case for innner product')
    except AttributeError:
      pass

    op_def = self.CommonConvert(op, 'FC')
    weight_tensor_name = op.name + '_weight:0'
    if op.data[0].ndim not in [2, 4]:
      raise ValueError('Unexpected weigth ndim.')
    if op.data[0].ndim == 4 and list(op.data[0].shape[:2]) != [1, 1]:
      raise ValueError('Do not support 4D weight with shape [1, 1, *, *]')
    input_shape = op.get_single_parent().output_shape_map[op.layer.bottom[0]]

    weight_data = op.data[0].reshape(-1, op.data[0].shape[-1])
    assert weight_data.shape[1] == (input_shape[1] * input_shape[2] * input_shape[3])
    weight_data = weight_data.reshape(-1, input_shape[3], input_shape[1], input_shape[2])
    weight_data = weight_data.transpose((0, 2, 3, 1)).reshape(weight_data.shape[0], -1)
    self.add_tensor(weight_tensor_name, weight_data)
    if self.device == 'gpu':
      if (weight_data.shape[0] + 3) / 4 > OPENCL_IMAGE_MAX_SIZE \
          or weight_data.shape[1] > OPENCL_IMAGE_MAX_SIZE:
        raise Exception('Mace gpu do not support FC with weight shape: '
                        +str(weight_data.shape))
      buffer_type = "WEIGHT_HEIGHT"
      output_name = self.add_buffer_to_image(weight_tensor_name, buffer_type)
      op_def.input.extend([output_name])
    else:
      op_def.input.extend([weight_tensor_name])

    # Add Bias
    if len(op.data) == 2:
      bias_tensor_name = op.name + '_bias:0'
      bias_data = op.data[1].reshape(-1)
      self.add_tensor(bias_tensor_name, bias_data)
      if self.device == 'gpu':
        output_name = self.add_buffer_to_image(bias_tensor_name, "ARGUMENT")
        op_def.input.extend([output_name])
      else:
        op_def.input.extend([bias_tensor_name])

    self.resolved_ops.add(op.name)
    output_shape = Shapes.fully_connected_shape(input_shape, weight_data.shape)
    op.output_shape_map[op.layer.top[0]] = output_shape
    final_op = op

    if len(self.ops_map[final_op.name].children) == 1 \
        and self.ops_map[final_op.name].children[0].type in activation_name_map:
      activation_op = self.ops_map[final_op.name].children[0]
      fused_act_arg = op_def.arg.add()
      fused_act_arg.name = 'activation'
      fused_act_arg.s = activation_name_map[activation_op.type]
      final_op = activation_op
      final_op.output_shape_map[final_op.layer.top[0]] = output_shape
      self.resolved_ops.add(activation_op.name)

    op_def.output.extend([final_op.name + ':0'])
    self.add_output_shape(op_def, output_shape)
    self.net_def.op.extend([op_def])

  def convert_pooling(self, op):
    op_def = self.CommonConvert(op, 'Pooling')

    param = op.layer.pooling_param
    paddings, strides, kernels = self.add_stride_pad_kernel_arg(param, op_def)
    if param.pool == caffe_pb2.PoolingParameter.MAX:
      pooling_type = "MaxPool"
    elif param.pool == caffe_pb2.PoolingParameter.AVE:
      pooling_type = "AvgPool"
    pooling_type_arg = op_def.arg.add()
    pooling_type_arg.name = 'pooling_type'
    pooling_type_arg.i = pooling_type_mode[pooling_type]

    input_shape = op.get_single_parent().output_shape_map[op.layer.bottom[0]]
    filter_shape = [kernels[0], kernels[1], input_shape[3], input_shape[3]]
    output_shape = Shapes.conv_pool_shape(input_shape, filter_shape,
                                          paddings, strides, [1, 1], math.ceil)
    op.output_shape_map[op.layer.top[0]] = output_shape

    op_def.output.extend([op.name + ':0'])
    self.add_output_shape(op_def, output_shape)
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_activation(self, op):
    op_def = self.CommonConvert(op, 'Activation')
    activation_arg = op_def.arg.add()
    activation_arg.name = 'activation'
    activation_arg.s = activation_name_map[op.type]
    op_def.output.extend([op.name + ':0'])
    output_shape = op.get_single_parent().output_shape_map[op.layer.bottom[0]]
    op.output_shape_map[op.layer.top[0]] = output_shape
    self.add_output_shape(op_def, output_shape)
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_prelu(self, op):
    op_def = self.CommonConvert(op, 'Activation')
    activation_arg = op_def.arg.add()
    activation_arg.name = 'activation'
    activation_arg.s = 'PRELU'
    alpha_tensor_name = op.name + '_alpha:0'
    alpha_data = op.data[0].reshape(-1)
    self.add_tensor(alpha_tensor_name, alpha_data)
    if self.device == 'gpu':
      output_name = self.add_buffer_to_image(alpha_tensor_name, "ARGUMENT")
      op_def.input.extend([output_name])
    else:
      op_def.input.extend([alpha_tensor_name])
    op_def.output.extend([op.name + ':0'])
    output_shape = op.get_single_parent().output_shape_map[op.layer.bottom[0]]
    op.output_shape_map[op.layer.top[0]] = output_shape
    self.add_output_shape(op_def, output_shape)
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_add(self, op):
    op_def = self.CommonConvert(op, 'AddN')
    op_def.output.extend([op.name + ':0'])
    output_shape = op.parents[0].output_shape_map[op.layer.bottom[0]]
    op.output_shape_map[op.layer.top[0]] = output_shape
    self.add_output_shape(op_def, output_shape)
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_concat(self, op):
    op_def = self.CommonConvert(op, 'Concat')
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

    input_shapes = []
    for i in range(len(op.parents)):
      input_shapes.append(op.parents[i].output_shape_map[op.layer.bottom[i]])
    output_shape = Shapes.concat_shape(input_shapes, axis_arg.i)
    op.output_shape_map[op.layer.top[0]] = output_shape
    self.add_output_shape(op_def, output_shape)
    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_eltwise(self, op):
    op_def = self.CommonConvert(op, 'Eltwise')
    param = op.layer.eltwise_param
    type_arg = op_def.arg.add()
    type_arg.name = 'type'
    type_arg.i = param.operation
    if len(param.coeff) > 0:
      coeff_arg = op_def.arg.add()
      coeff_arg.name = 'coeff'
      coeff_arg.ints.extend(list(param.coeff))

    output_shape = op.parents[0].output_shape_map[op.layer.bottom[0]]
    op.output_shape_map[op.layer.top[0]] = output_shape
    self.add_output_shape(op_def, output_shape)
    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_slice(self, op):
    op_def = self.CommonConvert(op, 'Slice')
    if op.layer.HasField('slice_param'):
      param = op.layer.slice_param
      if param.HasField('axis') and param.axis != 1:
        raise Exception('Mace do not support slice with axis ' + str(param.axis))
      if len(param.slice_point) > 0:
        raise Exception('Mace do not support slice with slice_point')

    input_shape = op.parents[0].output_shape_map[op.layer.bottom[0]]
    num_outputs = len(op.layer.top)
    if (input_shape[3] % num_outputs) != 0 or \
      (self.device == 'gpu' and ((input_shape[3] / num_outputs) % 4 != 0)) :
      raise Exception('Mace do not support slice with input shape '
                      + str(input_shape) + ' and number of output ' + str(num_outputs))
    output_shape = Shapes.slice_shape(input_shape, num_outputs)
    for i in range(len(op.layer.top)):
      op.output_shape_map[op.layer.top[i]] = output_shape
      self.add_output_shape(op_def, output_shape)
      op_def.output.extend([op.name + '_' + str(i) + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def convert_normal_op(self, op):
    op_def = self.CommonConvert(op, op.type)
    output_shape = op.parents[0].output_shape_map[op.layer.bottom[0]]
    op.output_shape_map[op.layer.top[0]] = output_shape
    self.add_output_shape(op_def, output_shape)
    op_def.output.extend([op.name + ':0'])
    self.net_def.op.extend([op_def])
    self.resolved_ops.add(op.name)

  def replace_in_out_name(self, input_names, output_names, is_single):
    in_names = set([input_name + ":0" for input_name in input_names])
    out_names = set([output_name + ":0" for output_name in output_names])
    if is_single:
      for op in self.net_def.op:
        if len(op.input) > 0 and op.input[0] in in_names:
          op.input[0] = MACE_INPUT_NODE_NAME + ':0'
        if len(op.output) > 0 and op.output[0] in out_names:
          op.output[0] = MACE_OUTPUT_NODE_NAME + ':0'
    else:
      for op in self.net_def.op:
        if len(op.input) > 0 and op.input[0] in in_names:
          op.input[0] = MACE_INPUT_NODE_NAME + '_' + op.input[0]
        if len(op.output) > 0 and op.output[0] in out_names:
          op.output[0] = MACE_OUTPUT_NODE_NAME + '_' + op.output[0]

  def add_input_op_shape(self, input_nodes, input_shapes):
    assert len(input_nodes) == len(input_shapes)
    for i in range(len(input_nodes)):
      input_op = self.ops_map[input_nodes[i]]
      if input_op.layer is not None:
        input_op.output_shape_map[input_op.layer.top[0]] = input_shapes[i]
      else:
        input_op.output_shape_map[input_op.name] = input_shapes[i]

  def convert(self, input_nodes, input_shapes, output_nodes):
    is_single = len(input_nodes) == 1 and len(output_nodes) == 1
    if self.device == 'gpu':
      self.add_input_transform(input_nodes, is_single)

    assert self.ops[0].type == 'Input'
    self.add_input_op_shape(input_nodes, input_shapes)

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
      elif op.type == 'Slice':
        self.convert_slice(op)
      else:
        raise Exception('Unknown Op: %s, type: %s' % (op.name, op.type))

    if self.device == 'gpu':
      self.add_output_transform(output_nodes, is_single)

    if self.device == 'cpu':
      self.replace_in_out_name(input_nodes, output_nodes, is_single)

    for op in self.ops:
      if op.name not in self.resolved_ops:
        print 'Unresolve Op: %s with type %s' % (op.name, op.type)


def convert_to_mace_pb(model_file, weight_file, input_node_str, input_shape_str, output_node_str, data_type, device, winograd):
  net_def = mace_pb2.NetDef()
  dt = data_type_map[data_type]

  caffe_net = caffe_pb2.NetParameter()
  with open(model_file, "r") as f:
    google.protobuf.text_format.Merge(str(f.read()), caffe_net)

  weights = caffe_pb2.NetParameter()
  with open(weight_file, "rb") as f:
    weights.MergeFromString(f.read())

  input_nodes = [x for x in input_node_str.split(',')]
  input_shapes = []
  if input_shape_str != "":
    input_shape_strs = [x for x in input_shape_str.split(':')]
    for shape_str in input_shape_strs:
      input_shapes.extend([[int(x) for x in shape_str.split(',')]])
  output_nodes = [x for x in output_node_str.split(',')]
  assert len(input_nodes) == len(input_shapes)

  converter = CaffeConverter(caffe_net, weights, net_def, dt, device, winograd)
  converter.convert(input_nodes, input_shapes, output_nodes)
  print "PB Converted."
  if device == 'gpu':
    print "start optimize memory."
    mem_optimizer = memory_optimizer.MemoryOptimizer(net_def)
    mem_optimizer.optimize()
    print "Memory optimization done."

  return net_def
