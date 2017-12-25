from mace.proto import mace_pb2
import tensorflow as tf
import numpy as np
from mace.python.tools import memory_optimizer

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
  'FILTER' : 0,
  'IN_OUT' : 1,
  'ARGUMENT' : 2,
}

data_type_map = {
  'DT_HALF' : mace_pb2.DT_HALF,
  'DT_FLOAT': mace_pb2.DT_FLOAT
}

BATCH_NORM_ORDER = ["Add", "Rsqrt", "Mul", "Mul", "Mul", "Sub", "Add"]

MACE_INPUT_NODE_NAME = "mace_input_node"
MACE_OUTPUT_NODE_NAME = "mace_output_node"

def get_input_tensor(op, index):
  input_tensor = op.inputs[index]
  if input_tensor.op.type == 'Reshape':
    input_tensor = get_input_tensor(input_tensor.op, 0)
  return input_tensor

class TFConverter(object):
  def __init__(self, tf_ops, net_def, dt, device):
    self.net_def = net_def
    self.tf_ops = tf_ops
    self.dt = dt
    self.device = device
    self.tf_graph = {}
    self.resolved_ops = {}

    for op in tf_ops:
      self.resolved_ops[op.name] = 0
      for input in op.inputs:
        input_name = input.name[:-2]
        if input_name not in self.tf_graph:
          self.tf_graph[input_name] = []
        self.tf_graph[input_name].append(op)

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

  def add_input_transform(self, name):
    new_input_name = MACE_INPUT_NODE_NAME + ":0"
    op_def = self.net_def.op.add()
    op_def.name = name
    op_def.type = 'BufferToImage'
    op_def.input.extend([new_input_name])
    op_def.output.extend([name+':0'])

    epsilon_arg = op_def.arg.add()
    epsilon_arg.name = 'buffer_type'
    epsilon_arg.i = buffer_type_map['IN_OUT']

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
    epsilon_arg.i = buffer_type_map['IN_OUT']

  @staticmethod
  def add_output_shape(outputs, op):
    output_shapes = []
    for output in outputs:
      if output.shape.num_elements() is not None:
        output_shape = mace_pb2.OutputShape()
        output_shape.dims.extend(output.shape.as_list())
        output_shapes.append(output_shape)
    op.output_shape.extend(output_shapes)

  def convert_tensor(self, op):
    tensor = self.net_def.tensors.add()
    tf_tensor = op.outputs[0].eval()
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
    if self.device == 'gpu':
      op_def.input.extend([op.inputs[0].name])
      output_name = self.add_buffer_to_image(op.inputs[1].name, "FILTER")
      op_def.input.extend([output_name])
    else:
      op_def.input.extend([input.name for input in op.inputs])

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
        output_name = self.add_buffer_to_image(bias_add_op.inputs[1].name, "ARGUMENT")
        op_def.input.extend([output_name])
      else:
        op_def.input.extend([bias_add_op.inputs[1].name])
      final_op = bias_add_op
      self.resolved_ops[bias_add_op.name] = 1

    if len(self.tf_graph[final_op.name]) == 1 \
        and self.tf_graph[final_op.name][0].type == 'Relu':
      relu_op = self.tf_graph[final_op.name][0]
      op_def.type = "FusedConv2D"
      final_op = relu_op
      self.resolved_ops[relu_op.name] = 1

    op_def.output.extend([output.name for output in final_op.outputs])
    self.add_output_shape(final_op.outputs, op_def)
    self.net_def.op.extend([op_def])

  def convert_fused_batchnorm(self, op):
    op_def = mace_pb2.OperatorDef()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = 'BatchNorm'
    if self.device == 'gpu':
      op_def.input.extend([op.inputs[0].name])
      for i in range(1, len(op.inputs)):
        output_name = self.add_buffer_to_image(op.inputs[i].name, "ARGUMENT")
        op_def.input.extend([output_name])
    else:
      op_def.input.extend([input.name for input in op.inputs])
    op_def.output.extend([op.outputs[0].name])

    self.add_output_shape(op.outputs, op_def)

    epsilon_arg = op_def.arg.add()
    epsilon_arg.name = 'epsilon'
    epsilon_arg.f = op.get_attr('epsilon')
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
    self.resolved_ops[op.name] = 1
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

  def convert_relu6(self, op):
    op_def = self.net_def.op.add()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = 'Relu'
    op_def.input.extend([input.name for input in op.inputs])
    op_def.output.extend([output.name for output in op.outputs])
    self.add_output_shape(op.outputs, op_def)
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
    op_def.input.extend([op.inputs[i].name for i in xrange(2)])
    op_def.output.extend([output.name for output in op.outputs])
    axis_arg = op_def.arg.add()
    axis_arg.name = 'axis'
    axis_arg.i = get_input_tensor(op, 2).eval().astype(np.int32)
    self.add_output_shape(op.outputs, op_def)
    self.resolved_ops[op.name] = 1

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

  def convert_bias_add(self, op):
    op_def = mace_pb2.OperatorDef()
    arg = op_def.arg.add()
    arg.name = 'T'
    arg.i = self.dt
    op_def.name = op.name
    op_def.type = "BiasAdd"
    op_def.input.extend([op.inputs[0].name])
    if self.device == 'gpu':
      output_name = self.add_buffer_to_image(op.inputs[1].name, "ARGUMENT")
      op_def.input.extend([output_name])
    else:
      op_def.input.extend([op.inputs[1].name])
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

  def convert(self, input_node, output_node):
    if self.device == 'gpu':
      self.add_input_transform(input_node)

    for op in self.tf_ops:
      if self.resolved_ops[op.name] == 1:
        continue
      if op.type in ['Placeholder', 'Reshape', 'Identity']:
        self.resolved_ops[op.name] = 1
        pass
      elif op.type == 'Const':
        self.convert_tensor(op)
      elif op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative':
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
      elif op.type in ['Relu']:
        self.convert_normal_op(op)
      else:
        raise Exception('Unknown Op: %s, type: %s' % (op.name, op.type))

    if self.device == 'gpu':
      self.add_output_transform(output_node)

    for key in self.resolved_ops:
      if self.resolved_ops[key] != 1:
        print 'Unresolve Op: %s' % key

def convert_to_mace_pb(input_graph_def, input_node, output_node, data_type, device):
  net_def = mace_pb2.NetDef()
  dt = data_type_map[data_type]

  with tf.Session() as session:
    with session.graph.as_default() as graph:
      tf.import_graph_def(input_graph_def, name="")
      ops = graph.get_operations()
      converter = TFConverter(ops, net_def, dt, device)
      converter.convert(input_node, output_node)
      print "PB Converted, start optimize memory."
      mem_optimizer = memory_optimizer.MemoryOptimizer(net_def)
      mem_optimizer.optimize()
      print "Memory optimization done."

  return net_def
