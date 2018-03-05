from lib.proto import mace_pb2
import tensorflow as tf
import numpy as np
import math
import copy
from tensorflow import gfile
from lib.python.tools import memory_optimizer
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2

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
}

data_type_map = {
  'DT_HALF' : mace_pb2.DT_HALF,
  'DT_FLOAT': mace_pb2.DT_FLOAT
}

activation_name_map = {
  'Relu' : 'RELU',
  'Sigmoid' : 'SIGMOID',
  'Tanh' : 'TANH',
  'Relu6' : 'RELUX'
}

BATCH_NORM_ORDER = ["Add", "Rsqrt", "Mul", "Mul", "Mul", "Sub", "Add"]

MACE_INPUT_NODE_NAME = "mace_input_node"
MACE_OUTPUT_NODE_NAME = "mace_output_node"

OPENCL_IMAGE_MAX_SIZE = 16384

def get_input_tensor(op, index):
  input_tensor = op.inputs[index]
  if input_tensor.op.type == 'Reshape':
    input_tensor = get_input_tensor(input_tensor.op, 0)
  return input_tensor

class TFConverter(object):
  def __init__(self, tf_ops, net_def, dt, device, winograd):
    self.net_def = net_def
    self.tf_ops = tf_ops
    self.dt = dt
    self.device = device
    self.winograd = winograd
    self.tf_graph = {}
    self.tf_parents = {}
    self.resolved_ops = {}
    self.unused_tensor = set()
    self.transpose_filter_tensor = {}
    self.reshape_tensor = {}
    self.ops = {}

    for op in tf_ops:
      self.ops[op.name] = op

    for op in tf_ops:
      self.resolved_ops[op.name] = 0
      for input in op.inputs:
        input_name = input.name[:-2]
        if input_name not in self.tf_graph:
          self.tf_graph[input_name] = []
        self.tf_graph[input_name].append(op)
        if op.name not in self.tf_parents:
          self.tf_parents[op.name] = []
        self.tf_parents[op.name].append(self.ops[input_name])

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
    op_def.output.extend([name+':0'])

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

  @staticmethod
  def add_output_shape(outputs, op):
    output_shapes = []
    for output in outputs:
      if output.shape.num_elements() is not None:
        output_shape = mace_pb2.OutputShape()
        output_shape.dims.extend(output.shape.as_list())
        output_shapes.append(output_shape)
    op.output_shape.extend(output_shapes)

  def add_tensor(self, name, shape, tf_dt, value):
    tensor = self.net_def.tensors.add()
    tensor.name = name

    shape = list(shape)
    tensor.dims.extend(shape)

    if tf_dt == tf.float32:
      tensor.data_type = mace_pb2.DT_FLOAT
      tensor.float_data.extend(value.flat)
    elif tf_dt == tf.int32:
      tensor.data_type = mace_pb2.DT_INT32
      tensor.int32_data.extend(value.flat)
    else:
      raise Exception("Not supported tensor type: " + tf_dt.name)

  def convert_reshape(self, op):
    input_tensor = get_input_tensor(op, 0)
    shape_tensor = get_input_tensor(op, 1)
    shape_value = shape_tensor.eval().astype(np.int32)
    self.unused_tensor.add(shape_tensor.name)
    self.reshape_tensor[input_tensor.name] = shape_value
    self.resolved_ops[op.name] = 1

  def convert_tensor(self, op):
    output_name = op.outputs[0].name
    if output_name not in self.unused_tensor:
      tensor = self.net_def.tensors.add()
      tf_tensor = op.outputs[0].eval()
      if output_name in self.transpose_filter_tensor:
        tf_tensor = tf_tensor.transpose(self.transpose_filter_tensor[output_name])
      if output_name in self.reshape_tensor:
        tf_tensor = tf_tensor.reshape(self.reshape_tensor[output_name])
      tensor.name = op.outputs[0].name

      shape = list(tf_tensor.shape)
      tensor.dims.extend(shape)

      tf_dt = op.get_attr('dtype')
      if tf_dt == tf.float32:
        tensor.data_type = mace_pb2.DT_FLOAT
        tensor.float_data.extend(tf_tensor.astype(np.float32).flat)
      elif tf_dt == tf.int32:
        tensor.data_type = mace_pb2.DT_INT32
        tensor.int32_data.extend(tf_tensor.astype(np.int32).flat)
      else:
        raise Exception("Not supported tensor type: " + tf_dt.name)
    self.resolved_ops[op.name] = 1

  def check_winograd_conv(self, op):
    filter_shape = get_input_tensor(op, 1).shape.as_list()
    strides = op.get_attr('strides')[1:3]
    output_shape = op.outputs[0].shape.as_list()
    if len(output_shape) == 0 or output_shape[0] is None:
      return False
    width = output_shape[0] * ((output_shape[1] + 1)/2) * ((output_shape[2]+1)/2)
    return self.winograd and op.type != 'DepthwiseConv2dNative' and self.device == 'gpu' and \
            filter_shape[0] == 3 and (filter_shape[0] == filter_shape[1]) and \
            (strides[0] == 1) and (strides[0] == strides[1]) and \
            (16 * filter_shape[2] < OPENCL_IMAGE_MAX_SIZE) and \
            (16 * filter_shape[3] < OPENCL_IMAGE_MAX_SIZE) and \
            (width < OPENCL_IMAGE_MAX_SIZE)

  def convert_winograd_conv(self, op):
    filter_tensor = get_input_tensor(op, 1)
    filter_shape = filter_tensor.shape.as_list()
    output_shape = op.outputs[0].shape.as_list()

    self.transpose_filter_tensor[filter_tensor.name] = (3, 2, 0, 1)
    filter_name = self.add_buffer_to_image(op.inputs[1].name, "WINOGRAD_FILTER")

    # Input transform
    wt_op = mace_pb2.OperatorDef()
    arg = wt_op.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    padding_arg = wt_op.arg.add()
    padding_arg.name = 'padding'
    padding_arg.i = padding_mode[op.get_attr('padding')]
    wt_op.name = op.name + '_input_transform'
    wt_op.type = 'WinogradTransform'
    wt_op.input.extend([op.inputs[0].name])
    wt_output_name = wt_op.name + ":0"
    wt_op.output.extend([wt_output_name])
    wt_output_shape = mace_pb2.OutputShape()
    wt_output_width = output_shape[0] * ((output_shape[1] + 1)/2) * ((output_shape[2]+1)/2)
    wt_output_shape.dims.extend([16, filter_shape[2], wt_output_width, 1])
    wt_op.output_shape.extend([wt_output_shape])

    # MatMul
    matmul_op = mace_pb2.OperatorDef()
    arg = matmul_op.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    matmul_op.name = op.name + '_matmul'
    matmul_op.type = 'MatMul'
    matmul_op.input.extend([filter_name, wt_output_name])
    matmul_output_name = matmul_op.name + ":0"
    matmul_op.output.extend([matmul_output_name])
    matmul_output_shape = mace_pb2.OutputShape()
    matmul_output_shape.dims.extend([16, filter_shape[3], wt_output_width, 1])
    matmul_op.output_shape.extend([matmul_output_shape])

    # Inverse transform
    iwt_op = mace_pb2.OperatorDef()
    arg = iwt_op.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    batch_arg = iwt_op.arg.add()
    batch_arg.name = 'batch'
    batch_arg.i = output_shape[0]
    height_arg = iwt_op.arg.add()
    height_arg.name = 'height'
    height_arg.i = output_shape[1]
    width_arg = iwt_op.arg.add()
    width_arg.name = 'width'
    width_arg.i = output_shape[2]
    iwt_op.name = op.name + '_inverse_transform'
    iwt_op.type = 'WinogradInverseTransform'
    iwt_op.input.extend([matmul_output_name])

    final_op = op
    self.resolved_ops[op.name] = 1

    if len(self.tf_graph[op.name]) == 1 and self.tf_graph[op.name][0].type == 'BiasAdd' :
      bias_add_op = self.tf_graph[op.name][0]
      output_name = self.add_buffer_to_image(get_input_tensor(bias_add_op, 1).name, "ARGUMENT")
      iwt_op.input.extend([output_name])
      final_op = bias_add_op
      self.resolved_ops[bias_add_op.name] = 1

    if len(self.tf_graph[final_op.name]) == 1 \
        and self.tf_graph[final_op.name][0].type in activation_name_map:
      activation_op = self.tf_graph[final_op.name][0]
      fused_act_arg = iwt_op.arg.add()
      fused_act_arg.name = 'activation'
      fused_act_arg.s = activation_name_map[activation_op.type]
      if activation_op.type == 'Relu6':
        max_limit_arg = iwt_op.arg.add()
        max_limit_arg.name = 'max_limit'
        max_limit_arg.f = 6
      final_op = activation_op
      self.resolved_ops[activation_op.name] = 1

    iwt_op.output.extend([output.name for output in final_op.outputs])
    self.add_output_shape(final_op.outputs, iwt_op)
    self.net_def.op.extend([wt_op, matmul_op, iwt_op])


  def convert_conv2d(self, op):
    op_def = mace_pb2.OperatorDef()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    if op.type == 'DepthwiseConv2dNative':
      op_def.type = 'DepthwiseConv2d'
    else:
      op_def.type = op.type
      self.transpose_filter_tensor[get_input_tensor(op, 1).name] = (0, 1, 3, 2)
    if self.device == 'gpu':
      op_def.input.extend([op.inputs[0].name])
      buffer_type = "DW_CONV2D_FILTER" if op_def.type == 'DepthwiseConv2d' else "CONV2D_FILTER"
      output_name = self.add_buffer_to_image(get_input_tensor(op, 1).name, buffer_type)
      op_def.input.extend([output_name])
    else:
      op_def.input.extend([get_input_tensor(op, i).name for i in range(len(op.inputs))])

    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding'
    padding_arg.i = padding_mode[op.get_attr('padding')]
    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    strides_arg.ints.extend(op.get_attr('strides')[1:3])
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
    final_op = op
    self.resolved_ops[op.name] = 1

    if len(self.tf_graph[op.name]) == 1 and self.tf_graph[op.name][0].type == 'BiasAdd' :
      bias_add_op = self.tf_graph[op.name][0]
      if self.device == 'gpu':
        output_name = self.add_buffer_to_image(get_input_tensor(bias_add_op, 1).name, "ARGUMENT")
        op_def.input.extend([output_name])
      else:
        op_def.input.extend([get_input_tensor(bias_add_op, 1).name])
      final_op = bias_add_op
      self.resolved_ops[bias_add_op.name] = 1

    if len(self.tf_graph.get(final_op.name, [])) == 1 \
        and self.tf_graph[final_op.name][0].type in activation_name_map:
      activation_op = self.tf_graph[final_op.name][0]
      op_def.type = "FusedConv2D"
      fused_act_arg = op_def.arg.add()
      fused_act_arg.name = 'activation'
      fused_act_arg.s = activation_name_map[activation_op.type]
      if activation_op.type == 'Relu6':
        max_limit_arg = op_def.arg.add()
        max_limit_arg.name = 'max_limit'
        max_limit_arg.f = 6
      final_op = activation_op
      self.resolved_ops[activation_op.name] = 1

    op_def.output.extend([output.name for output in final_op.outputs])
    self.add_output_shape(final_op.outputs, op_def)
    self.net_def.op.extend([op_def])

  def convert_fused_batchnorm(self, op):
    op_def = mace_pb2.OperatorDef()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
    op_def.name = op.name
    op_def.type = 'FoldedBatchNorm'

    gamma_tensor = get_input_tensor(op, 1)
    for i in range(1, 5):
      input_tensor = get_input_tensor(op, i)
      assert input_tensor.shape == gamma_tensor.shape
      self.unused_tensor.add(input_tensor.name)

    gamma_value = get_input_tensor(op, 1).eval().astype(np.float32)
    beta_value = get_input_tensor(op, 2).eval().astype(np.float32)
    mean_value = get_input_tensor(op, 3).eval().astype(np.float32)
    var_value = get_input_tensor(op, 4).eval().astype(np.float32)
    epsilon_value = op.get_attr('epsilon')

    scale_value = (
      (1.0 / np.vectorize(math.sqrt)(var_value + epsilon_value)) *
      gamma_value)
    offset_value = (-mean_value * scale_value) + beta_value
    idx = gamma_tensor.name.rfind('/')
    name_prefix = gamma_tensor.name[:idx] + '/'
    input_names = [name_prefix+'scale:0', name_prefix+'offset:0']
    self.add_tensor(input_names[0], gamma_value.shape,
      gamma_tensor.dtype, scale_value)
    self.add_tensor(input_names[1], gamma_value.shape,
      gamma_tensor.dtype, offset_value)

    op_def.input.extend([op.inputs[0].name])
    if self.device == 'gpu':
      for name in input_names:
        output_name = self.add_buffer_to_image(name, "ARGUMENT")
        op_def.input.extend([output_name])
    else:
      op_def.input.extend([name for name in input_names])

    self.resolved_ops[op.name] = 1
    final_op = op

    if len(self.tf_graph[op.name]) == 1 \
        and self.tf_graph[op.name][0].type in activation_name_map:
      activation_op = self.tf_graph[op.name][0]
      fused_act_arg = op_def.arg.add()
      fused_act_arg.name = 'activation'
      fused_act_arg.s = activation_name_map[activation_op.type]
      if activation_op.type == 'Relu6':
        max_limit_arg = op_def.arg.add()
        max_limit_arg.name = 'max_limit'
        max_limit_arg.f = 6
      final_op = activation_op
      self.resolved_ops[activation_op.name] = 1

    op_def.output.extend([final_op.outputs[0].name])
    self.add_output_shape([final_op.outputs[0]], op_def)

    self.net_def.op.extend([op_def])

  def convert_batchnorm(self, op):
    bn_ops = []
    bn_ops.append(op)
    for i in range(1, 3):
      if len(self.tf_graph[bn_ops[i-1].name]) == 1 \
          and self.tf_graph[bn_ops[i-1].name][0].type == BATCH_NORM_ORDER[i]:
        bn_ops.append(self.tf_graph[bn_ops[i-1].name][0])
      else:
        raise Exception('Invalid BatchNorm Op')
    if len(self.tf_graph[bn_ops[2].name]) == 2 \
        and self.tf_graph[bn_ops[2].name][0].type == BATCH_NORM_ORDER[3] \
        and self.tf_graph[bn_ops[2].name][1].type == BATCH_NORM_ORDER[4]:
      bn_ops.append(self.tf_graph[bn_ops[2].name][0])
      bn_ops.append(self.tf_graph[bn_ops[2].name][1])
    else:
      raise Exception('Invalid BatchNorm Op')
    bn_ops.append(self.tf_graph[bn_ops[4].name][0])
    bn_ops.append(self.tf_graph[bn_ops[3].name][0])

    op_def = mace_pb2.OperatorDef()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt

    input_name = get_input_tensor(bn_ops[3], 0).name
    gamma = get_input_tensor(bn_ops[2], 1).name
    beta = get_input_tensor(bn_ops[5], 0).name
    mean = get_input_tensor(bn_ops[4], 0).name
    variance = get_input_tensor(bn_ops[0], 0).name

    op_def.name = op.name[:-4]  # remove /add
    op_def.type = 'BatchNorm'
    if self.device == 'gpu':
      op_def.input.extend([input_name])
      for tensor_name in [gamma, beta, mean, variance]:
        output_name = self.add_buffer_to_image(tensor_name, "ARGUMENT")
        op_def.input.extend([output_name])
    else:
      op_def.input.extend([input_name, gamma, beta, mean, variance])
    op_def.output.extend([output.name for output in bn_ops[6].outputs])
    self.add_output_shape(bn_ops[6].outputs, op_def)
    epsilon_arg = op_def.arg.add()
    epsilon_arg.name = 'epsilon'
    epsilon_arg.f = get_input_tensor(op, 1).eval().astype(np.float)
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
    self.unused_tensor.add(get_input_tensor(op, 1).name)

    self.net_def.op.extend([op_def])
    for i in range(0, 7):
      self.resolved_ops[bn_ops[i].name] = 1

  def convert_pooling(self, op):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = 'Pooling'
    op_def.input.extend([input.name for input in op.inputs])
    op_def.output.extend([output.name for output in op.outputs])
    self.add_output_shape(op.outputs, op_def)
    pooling_type_arg = op_def.arg.add()
    pooling_type_arg.name = 'pooling_type'
    pooling_type_arg.i = pooling_type_mode[op.type]
    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding'
    padding_arg.i = padding_mode[op.get_attr('padding')]
    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    strides_arg.ints.extend(op.get_attr('strides')[1:3])
    kernels_arg = op_def.arg.add()
    kernels_arg.name = 'kernels'
    kernels_arg.ints.extend(op.get_attr('ksize')[1:3])
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
    self.resolved_ops[op.name] = 1

  def convert_global_avg_pooling(self, op):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = 'Pooling'
    op_def.input.extend([op.inputs[0].name])
    op_def.output.extend([output.name for output in op.outputs])
    self.add_output_shape(op.outputs, op_def)
    pooling_type_arg = op_def.arg.add()
    pooling_type_arg.name = 'pooling_type'
    pooling_type_arg.i = pooling_type_mode['AvgPool']
    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding'
    padding_arg.i = padding_mode['VALID']
    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    strides_arg.ints.extend([1, 1])
    kernels_arg = op_def.arg.add()
    kernels_arg.name = 'kernels'
    kernels_arg.ints.extend(op.inputs[0].shape.as_list()[1:3])
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
    self.resolved_ops[op.name] = 1

  def convert_activation(self, op):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = 'Activation'
    activation_arg = op_def.arg.add()
    activation_arg.name = 'activation'
    activation_arg.s = activation_name_map[op.type]
    op_def.input.extend([input.name for input in op.inputs])
    op_def.output.extend([output.name for output in op.outputs])
    self.add_output_shape(op.outputs, op_def)
    self.resolved_ops[op.name] = 1

  def convert_relu6(self, op):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = 'Activation'
    op_def.input.extend([input.name for input in op.inputs])
    op_def.output.extend([output.name for output in op.outputs])
    self.add_output_shape(op.outputs, op_def)
    activation_arg = op_def.arg.add()
    activation_arg.name = 'activation'
    activation_arg.s = "RELUX"
    max_limit_arg = op_def.arg.add()
    max_limit_arg.name = 'max_limit'
    max_limit_arg.f = 6
    self.resolved_ops[op.name] = 1

  def convert_add(self, op):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = "AddN"
    op_def.input.extend([input.name for input in op.inputs])
    op_def.output.extend([output.name for output in op.outputs])
    self.add_output_shape(op.outputs, op_def)
    self.resolved_ops[op.name] = 1

  def convert_concat(self, op):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = "Concat"
    op_def.input.extend([input.name for input in op.inputs[:-1]])
    op_def.output.extend([output.name for output in op.outputs])
    axis_arg = op_def.arg.add()
    axis_arg.name = 'axis'
    axis_arg.i = get_input_tensor(op, len(op.inputs) - 1).eval().astype(np.int32)
    self.add_output_shape(op.outputs, op_def)
    self.resolved_ops[op.name] = 1
    self.unused_tensor.add(get_input_tensor(op, len(op.inputs) - 1).name)

  def convert_resize_bilinear(self, op):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = "ResizeBilinear"
    op_def.input.extend([op.inputs[0].name])
    op_def.output.extend([output.name for output in op.outputs])
    size_arg = op_def.arg.add()
    size_arg.name = 'size'
    size_arg.ints.extend(get_input_tensor(op, 1).eval().astype(np.int32).flat)
    size_arg = op_def.arg.add()
    size_arg.name = 'align_corners'
    size_arg.i = op.get_attr('align_corners')
    self.add_output_shape(op.outputs, op_def)
    self.resolved_ops[op.name] = 1
    self.unused_tensor.add(get_input_tensor(op, 1).name)

  def convert_bias_add(self, op):
    op_def = mace_pb2.OperatorDef()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = "BiasAdd"
    op_def.input.extend([op.inputs[0].name])
    if self.device == 'gpu':
      output_name = self.add_buffer_to_image(get_input_tensor(op, 1).name, "ARGUMENT")
      op_def.input.extend([output_name])
    else:
      op_def.input.extend([get_input_tensor(op, 1).name])
    op_def.output.extend([output.name for output in op.outputs])
    self.add_output_shape(op.outputs, op_def)
    self.net_def.op.extend([op_def])
    self.resolved_ops[op.name] = 1

  def convert_space_to_batch(self, op, b2s):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = op.type
    op_def.input.extend([op.inputs[0].name])
    op_def.output.extend([output.name for output in op.outputs])
    size_arg = op_def.arg.add()
    size_arg.name = 'block_shape'
    size_arg.ints.extend(get_input_tensor(op, 1).eval().astype(np.int32).flat)
    size_arg = op_def.arg.add()
    if b2s:
      size_arg.name = 'crops'
    else:
      size_arg.name = 'paddings'
    size_arg.ints.extend(get_input_tensor(op, 2).eval().astype(np.int32).flat)
    self.add_output_shape(op.outputs, op_def)
    self.resolved_ops[op.name] = 1
    self.unused_tensor.add(get_input_tensor(op, 1).name)
    self.unused_tensor.add(get_input_tensor(op, 2).name)

  def is_atrous_conv2d(self, op):
    return op.type == 'SpaceToBatchND' and\
           len(self.tf_graph[op.name]) == 1 and self.tf_graph[op.name][0].type == 'Conv2D'

  def convert_atrous_conv2d(self, op):
    op_def = mace_pb2.OperatorDef()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    conv_op = self.tf_graph[op.name][0]
    op_def.name = conv_op.name
    op_def.type = conv_op.type
    self.transpose_filter_tensor[get_input_tensor(conv_op, 1).name] = (0, 1, 3, 2)
    if self.device == 'gpu':
      op_def.input.extend([op.inputs[0].name])
      output_name = self.add_buffer_to_image(get_input_tensor(conv_op, 1).name, "CONV2D_FILTER")
      op_def.input.extend([output_name])
    else:
      op_def.input.extend([get_input_tensor(op, 0).name])
      op_def.input.extend([get_input_tensor(conv_op, 1).name])

    dilation_arg = op_def.arg.add()
    dilation_arg.name = 'dilations'
    dilation_arg.ints.extend(get_input_tensor(op, 1).eval().astype(np.int32).flat)
    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding'
    padding_values = get_input_tensor(op, 2).eval().astype(np.int32).flat
    if len(padding_values) > 0 and padding_values[0] > 0:
      padding_arg.i = padding_mode['SAME']
    else:
      padding_arg.i = padding_mode['VALID']
    self.unused_tensor.add(get_input_tensor(op, 1).name)
    self.unused_tensor.add(get_input_tensor(op, 2).name)

    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    strides_arg.ints.extend([1, 1])
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
    final_op = conv_op
    self.resolved_ops[op.name] = 1
    self.resolved_ops[conv_op.name] = 1

    if len(self.tf_graph[final_op.name]) == 1 and self.tf_graph[final_op.name][0].type == 'BiasAdd' :
      bias_add_op = self.tf_graph[final_op.name][0]
      if self.device == 'gpu':
        output_name = self.add_buffer_to_image(get_input_tensor(bias_add_op, 1).name, "ARGUMENT")
        op_def.input.extend([output_name])
      else:
        op_def.input.extend([get_input_tensor(bias_add_op, 1).name])
      final_op = bias_add_op
      self.resolved_ops[bias_add_op.name] = 1

    if len(self.tf_graph[final_op.name]) == 1 \
      and self.tf_graph[final_op.name][0].type == 'BatchToSpaceND':
      final_op = self.tf_graph[final_op.name][0]
      self.resolved_ops[final_op.name] = 1
      self.unused_tensor.add(get_input_tensor(final_op, 1).name)
      self.unused_tensor.add(get_input_tensor(final_op, 2).name)
    else:
      raise Exception('Convert atrous conv error: no BatchToSpaceND op')

    if len(self.tf_graph[final_op.name]) == 1 \
        and self.tf_graph[final_op.name][0].type == 'Relu':
      relu_op = self.tf_graph[final_op.name][0]
      op_def.type = "FusedConv2D"
      fused_relu_arg = op_def.arg.add()
      fused_relu_arg.name = 'activation'
      fused_relu_arg.s = "RELU"
      final_op = relu_op
      self.resolved_ops[relu_op.name] = 1

    op_def.output.extend([output.name for output in final_op.outputs])
    self.add_output_shape(final_op.outputs, op_def)
    self.net_def.op.extend([op_def])

  def is_softmax(self, op):
    return op.type == 'Softmax' and \
           len(self.tf_parents[op.name]) == 1 and self.tf_parents[op.name][0].type == 'Reshape' and \
           len(self.tf_graph[op.name]) == 1 and self.tf_graph[op.name][0].type == 'Reshape'

  def convert_softmax(self, softmax_op):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt

    # deal with first Reshape op
    parent_reshape_op = self.tf_parents[softmax_op.name][0]
    self.unused_tensor.add(get_input_tensor(parent_reshape_op, 1).name)
    self.resolved_ops[parent_reshape_op.name] = 1

    # FIXME: hardcode for inception_v3
    # remove squeeze if exist
    squeeze_op = self.tf_parents[parent_reshape_op.name][0]
    if squeeze_op.type == 'Squeeze':
      op_def.input.extend([squeeze_op.inputs[0].name])
      self.resolved_ops[squeeze_op.name] = 1
      # remove shape if exist
      children_ops = self.tf_graph[squeeze_op.name]
      print children_ops
      if len(children_ops) > 1 and children_ops[0].type == 'Shape':
        self.unused_tensor.add(get_input_tensor(children_ops[1], 0).name)
        self.resolved_ops[children_ops[1].name] = 1
    else:
      op_def.input.extend([parent_reshape_op.inputs[0].name])

    # deal with Softmax op
    op_def.name = softmax_op.name
    op_def.type = softmax_op.type
    self.resolved_ops[softmax_op.name] = 1

    # deal with last Reshape op
    reshape_op = self.tf_graph[softmax_op.name][0]
    self.unused_tensor.add(get_input_tensor(reshape_op, 1).name)

    if reshape_op.outputs[0].shape.ndims == 2:
      shape = reshape_op.outputs[0].shape
      from tensorflow.python.framework.tensor_shape import as_shape
      reshape_op.outputs[0]._shape = as_shape([1, 1, shape[0], shape[1]])
    op_def.output.extend([output.name for output in reshape_op.outputs])
    self.add_output_shape(reshape_op.outputs, op_def)
    self.resolved_ops[reshape_op.name] = 1

  def convert_normal_op(self, op):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = op.type
    op_def.input.extend([input.name for input in op.inputs])
    op_def.output.extend([output.name for output in op.outputs])
    self.add_output_shape(op.outputs, op_def)
    self.resolved_ops[op.name] = 1

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

    for op in self.tf_ops:
      if self.resolved_ops[op.name] == 1:
        continue
      if op.type in ['Placeholder', 'Identity']:
        self.resolved_ops[op.name] = 1
        pass
      elif op.type == 'Const':
        pass
      elif op.type == 'Reshape':
        self.convert_reshape(op)
      elif self.is_atrous_conv2d(op):
        self.convert_atrous_conv2d(op)
      elif op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative':
        if self.check_winograd_conv(op):
          self.convert_winograd_conv(op)
        else:
          self.convert_conv2d(op)
      elif op.type == 'FusedBatchNorm':
        self.convert_fused_batchnorm(op)
      elif op.type == 'Add' and op.name.endswith('batchnorm/add'):
        self.convert_batchnorm(op)
      elif op.type == 'AvgPool' or op.type == 'MaxPool':
        self.convert_pooling(op)
      elif op.type == 'Relu6':
        self.convert_relu6(op)
      elif op.type == 'Add':
        self.convert_add(op)
      elif op.type == 'ConcatV2':
        self.convert_concat(op)
      elif op.type == 'ResizeBilinear':
        self.convert_resize_bilinear(op)
      elif op.type == 'BiasAdd':
        self.convert_bias_add(op)
      elif op.type == 'SpaceToBatchND':
        self.convert_space_to_batch(op, False)
      elif op.type == 'BatchToSpaceND':
        self.convert_space_to_batch(op, True)
      elif self.is_softmax(op):
        self.convert_softmax(op)
      elif op.type in ['Relu', 'Sigmoid', 'Tanh']:
        self.convert_activation(op)
      # FIXME: hardcode for inception_v3
      elif op.type in ['Squeeze', 'Shape']:
        self.resolved_ops[op.name] = 1
      elif op.type == 'Mean':
        # Global avg pooling
        reduce_dims = op.inputs[1].eval()
        if reduce_dims[0] == 1 and reduce_dims[1] == 2:
          self.convert_global_avg_pooling(op)
          self.unused_tensor.add(op.inputs[1].name)
        else:
          raise Exception('Unknown Op: %s, type: %s' % (op.name, op.type))
      #elif op.type in ['']:
      #  self.convert_normal_op(op)
      else:
        raise Exception('Unknown Op: %s, type: %s' % (op.name, op.type))

    for op in self.tf_ops:
      if self.resolved_ops[op.name] == 1:
        continue
      elif op.type == 'Const':
        self.convert_tensor(op)
      else:
        raise Exception('Unknown Op: %s, type: %s' % (op.name, op.type))

    if self.device == 'gpu':
      self.add_output_transform(output_node)

    if self.device == 'cpu':
      self.replace_in_out_name(input_node, output_node)

    for key in self.resolved_ops:
      if self.resolved_ops[key] != 1:
        print 'Unresolve Op: %s' % key

class Optimizer:
  def __init__(self, net_def, device):
    self.net_def = net_def
    self.device = device
    self.mace_graph = {}
    self.tensor_map = {}
    for op in net_def.op:
      for input_name in op.input:
        if input_name not in self.mace_graph:
          self.mace_graph[input_name] = []
        self.mace_graph[input_name].append(op)

    for tensor in net_def.tensors:
      self.tensor_map[tensor.name] = tensor

  def get_buffer_tensor_name(self, name):
    if self.device == 'gpu':
      return name[:-6] + name[-2:]
    else:
      return name

  def fold_batch_norm(self):
    unused_tensors = set()
    new_tensors = []
    new_net = mace_pb2.NetDef()
    resolved_ops = set()

    for op in self.net_def.op:
      if op.name in resolved_ops:
        pass
      elif op.type == 'DepthwiseConv2d' and len(op.output) == 1 \
          and self.mace_graph[op.output[0]][0].type == 'FoldedBatchNorm':
        depthwise_conv2d_op = op
        folded_bn_op = self.mace_graph[op.output[0]][0]
        weight_buffer_name = self.get_buffer_tensor_name(depthwise_conv2d_op.input[1])
        weight_tensor = self.tensor_map[weight_buffer_name]
        scale_buffer_name = self.get_buffer_tensor_name(folded_bn_op.input[1])
        offset_buffer_name = self.get_buffer_tensor_name(folded_bn_op.input[2])
        scale_tensor = self.tensor_map[scale_buffer_name]
        weight_shape = weight_tensor.dims
        idx = 0
        for i in range(weight_shape[0]):
          for j in range(weight_shape[1]):
            for ic in range(weight_shape[2]):
              for oc in range(weight_shape[3]):
                weight_tensor.float_data[idx] *= scale_tensor.float_data[ic * weight_shape[3] + oc]
                idx += 1

        new_tensors.append(weight_tensor)
        unused_tensors.add(weight_tensor.name)
        unused_tensors.add(scale_tensor.name)

        if self.device == 'gpu':
          scale_b2i_op = self.mace_graph[scale_buffer_name][0]
          offset_b2i_op = self.mace_graph[offset_buffer_name][0]
          resolved_ops.add(scale_b2i_op.name)
          resolved_ops.add(offset_b2i_op.name)
          new_net.op.extend([offset_b2i_op])

        resolved_ops.add(depthwise_conv2d_op.name)
        resolved_ops.add(folded_bn_op.name)

        offset_tensor_name = folded_bn_op.input[2]
        depthwise_conv2d_op.input.extend([offset_tensor_name])

        for arg in folded_bn_op.arg:
          if arg.name == 'activation':
            act_arg = depthwise_conv2d_op.arg.add()
            act_arg.name = arg.name
            act_arg.s = arg.s
          elif arg.name == 'max_limit':
            act_arg = depthwise_conv2d_op.arg.add()
            act_arg.name = arg.name
            act_arg.f = arg.f

        depthwise_conv2d_op.output[0] = folded_bn_op.output[0]
        new_net.op.extend([depthwise_conv2d_op])
      else:
        new_net.op.extend([op])

    for tensor in self.net_def.tensors:
      if tensor.name in unused_tensors:
        pass
      else:
        new_net.tensors.extend([tensor])

    for tensor in new_tensors:
      new_net.tensors.extend([tensor])

    return new_net

  def optimize(self):
    new_net = self.fold_batch_norm()
    return new_net

def add_shape_info(input_graph_def, input_node, input_shape):
  inputs_replaced_graph = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node.name == input_node:
      placeholder_node = copy.deepcopy(node)
      placeholder_node.attr.clear()
      placeholder_node.attr['shape'].shape.dim.extend([
        tensor_shape_pb2.TensorShapeProto.Dim(size=i) for i in input_shape
      ])
      placeholder_node.attr['dtype'].CopyFrom(node.attr['dtype'])
      inputs_replaced_graph.node.extend([placeholder_node])
    else:
      inputs_replaced_graph.node.extend([copy.deepcopy(node)])
  return inputs_replaced_graph


def convert_to_mace_pb(model_file, input_node, input_shape, output_node, data_type, device, winograd):
  net_def = mace_pb2.NetDef()
  dt = data_type_map[data_type]

  input_graph_def = tf.GraphDef()
  with gfile.Open(model_file, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

  input_graph_def = add_shape_info(input_graph_def, input_node, input_shape)
  with tf.Session() as session:
    with session.graph.as_default() as graph:
      tf.import_graph_def(input_graph_def, name="")
      ops = graph.get_operations()
      converter = TFConverter(ops, net_def, dt, device, winograd)
      converter.convert(input_node, output_node)
      optimizer = Optimizer(net_def, device)
      net_def = optimizer.optimize()
      print "Model Converted."
      if device == 'gpu':
        print "start optimize memory."
        mem_optimizer = memory_optimizer.MemoryOptimizer(net_def)
        mem_optimizer.optimize()
        print "Memory optimization done."

  return net_def
