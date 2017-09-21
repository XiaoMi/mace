from mace.proto import mace_pb2
import tensorflow as tf
import numpy as np

padding_mode = {
  'VALID': 0,
  'SAME': 1,
  'FULL': 2
}
pooling_type_mode = {
  'AvgPool': 1,
  'MaxPool': 2
}


def convert_ops(unresolved_ops, net_def):
  ops_count = len(unresolved_ops)
  resolved_count = 1

  first_op = unresolved_ops[0]

  if first_op.type == 'Placeholder':
    pass
  elif first_op.type == 'Const':
    tf_tensor = first_op.outputs[0].eval()
    tensor = net_def.tensors.add()
    tensor.name = first_op.outputs[0].name
    # TODO: support other type than float
    tensor.data_type = mace_pb2.DT_FLOAT

    shape = list(tf_tensor.shape)
    if (first_op.name.find('pointwise_kernel') != -1 or
        first_op.name.find('depthwise_kernel') != -1 or
        first_op.name.endswith('weights') or
        first_op.name.endswith('kernel')) \
        and first_op.outputs[0].consumers()[0].type.find('Conv') != -1:
      tf_tensor = np.transpose(tf_tensor, axes=(3, 2, 0, 1))
      shape = [shape[3], shape[2], shape[0], shape[1]]
      # print (tensor.name, shape)
    tensor.dims.extend(shape)
    tensor.float_data.extend(tf_tensor.astype(float).flat)
  elif first_op.type == 'Conv2D' or first_op.type == 'DepthwiseConv2dNative':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    if first_op.type == 'DepthwiseConv2dNative':
      op_def.type = 'DepthwiseConv2d'
    else:
      op_def.type = first_op.type
    op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([output.name for output in first_op.outputs])
    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding'
    padding_arg.i = padding_mode[first_op.get_attr('padding')]
    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    strides_arg.ints.extend(first_op.get_attr('strides')[2:])
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = first_op.get_attr('data_format')
    if first_op.get_attr('data_format') != 'NCHW':
      raise Exception('only support NCHW now')

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
    if mul_op.type != 'Mul' or mul_2_op.type != 'Mul' or mul_1_op.type != 'Mul' or sub_op.type != 'Sub' or add_1_op.type != 'Add':
      raise Exception('Invalid BatchNorm Op')

    input_name = mul_1_op.inputs[0].name
    gamma = mul_op.inputs[1].name
    beta = sub_op.inputs[0].name
    mean = mul_2_op.inputs[0].name
    variance = add_op.inputs[0].name
    epsilon = add_op.inputs[1].name

    op_def = net_def.op.add()
    op_def.name = first_op.name[:-4]  # remove /add
    op_def.type = 'BatchNorm'
    op_def.input.extend([input_name, gamma, beta, mean, variance, epsilon])
    op_def.output.extend([output.name for output in add_1_op.outputs])

    resolved_count = 7
  elif first_op.type == 'Relu6':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = 'Relu'
    op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([output.name for output in first_op.outputs])
    max_limit_arg = op_def.arg.add()
    max_limit_arg.name = 'max_limit'
    max_limit_arg.f = 6
  elif first_op.type == 'Relu':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = first_op.type
    op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([output.name for output in first_op.outputs])
  elif first_op.type == 'AvgPool' or first_op.type == 'MaxPool':
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = 'Pooling'
    op_def.input.extend([input.name for input in first_op.inputs])
    op_def.output.extend([output.name for output in first_op.outputs])
    pooling_type_arg = op_def.arg.add()
    pooling_type_arg.name = 'pooling_type'
    pooling_type_arg.i = pooling_type_mode[first_op.type]
    padding_arg = op_def.arg.add()
    padding_arg.name = 'padding'
    padding_arg.i = padding_mode[first_op.get_attr('padding')]
    strides_arg = op_def.arg.add()
    strides_arg.name = 'strides'
    strides_arg.ints.extend(first_op.get_attr('strides')[2:])
    kernels_arg = op_def.arg.add()
    kernels_arg.name = 'kernels'
    kernels_arg.ints.extend(first_op.get_attr('ksize')[2:])
    data_format_arg = op_def.arg.add()
    data_format_arg.name = 'data_format'
    data_format_arg.s = first_op.get_attr('data_format')
    if first_op.get_attr('data_format') != 'NCHW':
      raise Exception('only support NCHW now')
  else:
    raise Exception('Unknown Op: ' + first_op.name)
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

  return net_def
