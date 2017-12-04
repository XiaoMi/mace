from mace.proto import mace_pb2
import tensorflow as tf
import numpy as np

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

def convert_tensor(op, tensor):
  tf_tensor = op.outputs[0].eval()
  tensor.name = op.outputs[0].name

  shape = list(tf_tensor.shape)
  tensor.dims.extend(shape)

  tf_dt = op.get_attr('dtype')
  if tf_dt == tf.float32:
    tensor.data_type = mace_pb2.DT_FLOAT
    tensor.float_data.extend(tf_tensor.astype(float).flat)
  elif tf_dt == tf.int32:
    tensor.data_type = mace_pb2.DT_INT32
    tensor.int32_data.extend(tf_tensor.astype(np.int32).flat)
  else:
    raise Exception("Not supported tensor type: " + tf_dt.name)


def get_input_tensor(op, index):
  input_tensor = op.inputs[index]
  if input_tensor.op.type == 'Reshape':
    input_tensor = get_input_tensor(input_tensor.op, 0)
  return input_tensor

def convert_ops(unresolved_ops, net_def):
  ops_count = len(unresolved_ops)
  resolved_count = 1

  first_op = unresolved_ops[0]

  if first_op.type in ['Placeholder', 'Reshape', 'Identity']:
    pass
  elif first_op.type == 'Const':
    tensor = net_def.tensors.add()
    convert_tensor(first_op, tensor)
  elif first_op.type == 'Conv2D' or first_op.type == 'DepthwiseConv2dNative':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    if first_op.type == 'DepthwiseConv2dNative':
      op_def.type = 'DepthwiseConv2d'
    else:
      op_def.type = first_op.type
    op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([output.name for output in first_op.outputs])
    output_shapes = []
    for output in first_op.outputs:
      output_shape = mace_pb2.OutputShape()
      output_shape.dims.extend(output.shape.as_list())
      output_shapes.append(output_shape)
    op_def.output_shape.extend(output_shapes)
    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding'
    padding_arg.i = padding_mode[first_op.get_attr('padding')]
    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    strides_arg.ints.extend(first_op.get_attr('strides')[1:3])
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'

    if ops_count >= 2 and unresolved_ops[1].type == 'BiasAdd':
      bias_add_op = unresolved_ops[1]
      op_def.input.extend([bias_add_op.inputs[1].name])
      resolved_count = 2
  elif first_op.type == 'Add' and first_op.name.endswith(
      'batchnorm/add') and ops_count > 7:
    add_op = first_op
    mul_op = unresolved_ops[2]
    mul_1_op = unresolved_ops[3]
    mul_2_op = unresolved_ops[4]
    sub_op = unresolved_ops[5]
    add_1_op = unresolved_ops[6]
    # print (mul_op.type, mul_2_op.type, mul_1_op.type, sub_op.type)
    if mul_op.type != 'Mul' or mul_2_op.type != 'Mul' or \
                    mul_1_op.type != 'Mul' or sub_op.type != 'Sub' or add_1_op.type != 'Add':
      raise Exception('Invalid BatchNorm Op')

    input_name = get_input_tensor(mul_1_op, 0).name
    gamma = get_input_tensor(mul_op, 1).name
    beta = get_input_tensor(sub_op, 0).name
    mean = get_input_tensor(mul_2_op, 0).name
    variance = get_input_tensor(add_op, 0).name
    epsilon = get_input_tensor(add_op, 1).name

    op_def = net_def.op.add()
    op_def.name = first_op.name[:-4]  # remove /add
    op_def.type = 'BatchNorm'
    op_def.input.extend([input_name, gamma, beta, mean, variance, epsilon])
    op_def.output.extend([output.name for output in add_1_op.outputs])
    output_shapes = []
    for output in add_1_op.outputs:
      output_shape = mace_pb2.OutputShape()
      output_shape.dims.extend(output.shape.as_list())
      output_shapes.append(output_shape)
    op_def.output_shape.extend(output_shapes)

    resolved_count = 7
  elif first_op.type == 'Relu6':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = 'Relu'
    op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([output.name for output in first_op.outputs])
    output_shapes = []
    for output in first_op.outputs:
      output_shape = mace_pb2.OutputShape()
      output_shape.dims.extend(output.shape.as_list())
      output_shapes.append(output_shape)
    op_def.output_shape.extend(output_shapes)
    max_limit_arg = op_def.arg.add()
    max_limit_arg.name = 'max_limit'
    max_limit_arg.f = 6
  elif first_op.type == 'AvgPool' or first_op.type == 'MaxPool':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = 'Pooling'
    op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([output.name for output in first_op.outputs])
    output_shapes = []
    for output in first_op.outputs:
      output_shape = mace_pb2.OutputShape()
      output_shape.dims.extend(output.shape.as_list())
      output_shapes.append(output_shape)
    op_def.output_shape.extend(output_shapes)
    pooling_type_arg = op_def.arg.add()
    pooling_type_arg.name = 'pooling_type'
    pooling_type_arg.i = pooling_type_mode[first_op.type]
    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding'
    padding_arg.i = padding_mode[first_op.get_attr('padding')]
    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    strides_arg.ints.extend(first_op.get_attr('strides')[1:3])
    kernels_arg = op_def.arg.add()
    kernels_arg.name = 'kernels'
    kernels_arg.ints.extend(first_op.get_attr('ksize')[1:3])
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
  elif first_op.type == 'Add':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = "AddN"
    op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([output.name for output in first_op.outputs])
    output_shapes = []
    for output in first_op.outputs:
      output_shape = mace_pb2.OutputShape()
      output_shape.dims.extend(output.shape.as_list())
      output_shapes.append(output_shape)
    op_def.output_shape.extend(output_shapes)
  elif first_op.type == 'ConcatV2':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = "Concat"
    op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([output.name for output in first_op.outputs])
    output_shapes = []
    for output in first_op.outputs:
      output_shape = mace_pb2.OutputShape()
      output_shape.dims.extend(output.shape.as_list())
      output_shapes.append(output_shape)
    op_def.output_shape.extend(output_shapes)
  elif first_op.type in ['Relu', 'ResizeBilinear', 'SpaceToBatchND',
                         'BatchToSpaceND', 'BiasAdd', 'FusedBatchNorm']:
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = first_op.type
    op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([output.name for output in first_op.outputs])
    output_shapes = []
    for output in first_op.outputs:
      output_shape = mace_pb2.OutputShape()
      output_shape.dims.extend(output.shape.as_list())
      output_shapes.append(output_shape)
    op_def.output_shape.extend(output_shapes)
  else:
    raise Exception('Unknown Op: %s, type: %s' % (first_op.name, first_op.type))
    pass

  for i in range(resolved_count):
    del unresolved_ops[0]


def convert_to_mace_pb(input_graph_def):
  net_def = mace_pb2.NetDef()

  with tf.Session() as session:
    with session.graph.as_default() as graph:
      tf.import_graph_def(input_graph_def, name="")
      ops = graph.get_operations()
      unresolved_ops = ops
      while len(unresolved_ops) > 0:
        convert_ops(unresolved_ops, net_def)

  print "Done."

  return net_def
