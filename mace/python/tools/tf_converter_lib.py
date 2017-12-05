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

buffer_type_map = {
  'FILTER' : 0,
  'IN_OUT' : 1,
  'ARGUMENT' : 2,
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

def add_buffer_to_image(input_name, input_type, net_def):
  output_name = input_name[:-2] + "_b2i" + input_name[-2:]
  op_def = net_def.op.add()
  op_def.name = output_name
  op_def.type = 'BufferToImage'
  op_def.input.extend([input_name])
  epsilon_arg = op_def.arg.add()
  epsilon_arg.name = 'buffer_type'
  epsilon_arg.i = buffer_type_map[input_type]
  epsilon_arg = op_def.arg.add()
  epsilon_arg.name = 'mode'
  epsilon_arg.i = 0
  return output_name

def convert_ops(unresolved_ops, net_def, device):
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
    if device == 'gpu':
      op_def.input.extend([first_op.inputs[0].name])
      output_name = add_buffer_to_image(first_op.inputs[1].name, "FILTER", net_def)
      op_def.input.extend([output_name])
    else:
      op_def.input.extend([input.name for input in first_op.inputs])

    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding'
    padding_arg.i = padding_mode[first_op.get_attr('padding')]
    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    strides_arg.ints.extend(first_op.get_attr('strides')[1:3])
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
    final_op = first_op

    if ops_count >= 3 and unresolved_ops[1].type == 'Const' and unresolved_ops[2].type == 'BiasAdd' :
      bias_tensor = unresolved_ops[1]
      tensor = net_def.tensors.add()
      convert_tensor(bias_tensor, tensor)

      bias_add_op = unresolved_ops[2]
      if device == 'gpu':
        output_name = add_buffer_to_image(bias_add_op.inputs[1].name, "ARGUMENT", net_def)
        op_def.input.extend([output_name])
      else:
        op_def.input.extend([bias_add_op.inputs[1].name])
      final_op = bias_add_op
      resolved_count = 3

    if ops_count >= 4 and unresolved_ops[3].type == 'Relu':
      relu_op = unresolved_ops[3];
      op_def.type = "FusedConv2D"
      final_op = relu_op
      resolved_count = 4

    op_def.output.extend([output.name for output in final_op.outputs])
    output_shapes = []
    for output in final_op.outputs:
      output_shape = mace_pb2.OutputShape()
      output_shape.dims.extend(output.shape.as_list())
      output_shapes.append(output_shape)
    op_def.output_shape.extend(output_shapes)

  elif first_op.type == 'FusedBatchNorm':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = 'BatchNorm'
    if device == 'gpu':
      op_def.input.extend([first_op.inputs[0].name])
      for i in range(1, len(first_op.inputs)):
        output_name = add_buffer_to_image(first_op.inputs[i].name, "ARGUMENT", net_def)
        op_def.input.extend([output_name])
    else:
      op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([first_op.outputs[0].name])

    output_shape = mace_pb2.OutputShape()
    output_shape.dims.extend(first_op.outputs[0].shape.as_list())
    op_def.output_shape.extend([output_shape])

    epsilon_arg = op_def.arg.add()
    epsilon_arg.name = 'epsilon'
    epsilon_arg.f = first_op.get_attr('epsilon')
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = 'NHWC'
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

    get_input_tensor(mul_1_op, 0)
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
    op_def.input.extend([first_op.inputs[i].name for i in xrange(2)])
    op_def.output.extend([output.name for output in first_op.outputs])
    axis_arg = op_def.arg.add()
    axis_arg.name = 'axis'
    axis_arg.i = get_input_tensor(first_op, 2).eval().astype(np.int32)
    output_shapes = []
    for output in first_op.outputs:
      output_shape = mace_pb2.OutputShape()
      output_shape.dims.extend(output.shape.as_list())
      output_shapes.append(output_shape)
    op_def.output_shape.extend(output_shapes)
  elif first_op.type == 'ResizeBilinear':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = "ResizeBilinear"
    op_def.input.extend([first_op.inputs[0].name])
    op_def.output.extend([output.name for output in first_op.outputs])
    size_arg = op_def.arg.add()
    size_arg.name = 'size'
    size_arg.ints.extend(get_input_tensor(first_op, 1).eval().astype(np.int32).flat)
    size_arg = op_def.arg.add()
    size_arg.name = 'align_corners'
    size_arg.ints.extend(first_op.get_attr('align_corners'))
    output_shapes = []
    for output in first_op.outputs:
      output_shape = mace_pb2.OutputShape()
      output_shape.dims.extend(output.shape.as_list())
      output_shapes.append(output_shape)
    op_def.output_shape.extend(output_shapes)
  elif first_op.type in ['Relu', 'SpaceToBatchND', 'BatchToSpaceND', 'BiasAdd']:
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


def convert_to_mace_pb(input_graph_def, device):
  net_def = mace_pb2.NetDef()

  with tf.Session() as session:
    with session.graph.as_default() as graph:
      tf.import_graph_def(input_graph_def, name="")
      ops = graph.get_operations()
      unresolved_ops = ops
      while len(unresolved_ops) > 0:
        convert_ops(unresolved_ops, net_def, device)

  print "PB Parsed."

  return net_def
