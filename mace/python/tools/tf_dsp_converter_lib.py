from mace.proto import mace_pb2
# import mace_pb2
import tensorflow as tf
import numpy as np
from operator import mul
from dsp_ops import DspOps
from mace.python.tools import tf_graph_util

padding_mode = {
  'NA': 0,
  'SAME': 1,
  'VALID': 2,
  'MIRROR_REFLECT': 3,
  'MIRROR_SYMMETRIC': 4,
  'SAME_CAFFE': 5
}

node_count = 0
node_ids = {}
resolved_ops = set()

def max_elem_size(tensor):
  if len(tensor.shape.as_list()) == 0:
    return tensor.dtype.size
  else:
    return reduce(mul, tensor.shape.as_list()) * tensor.dtype.size

def find_dtype(tensor_dtype):
  if tensor_dtype == tf.float32:
    return mace_pb2.DT_FLOAT
  elif tensor_dtype == tf.uint8 or tensor_dtype == tf.quint8:
    return mace_pb2.DT_UINT8
  elif tensor_dtype == tf.int32 or tensor_dtype == tf.qint32:
    return mace_pb2.DT_INT32
  else:
    raise Exception('Unsupported data type: ', tensor_dtype)

def has_padding_and_strides(op):
  return 'padding' in op.node_def.attr and 'strides' in op.node_def.attr

def is_node_flatten_reshape(op):
  return op.type == 'Reshape' and len(op.outputs[0].shape) == 1

def get_input_tensor(op, index):
  input_tensor = op.inputs[index]
  if input_tensor.op.type == 'Reshape':
    input_tensor = get_input_tensor(input_tensor.op, 0)
  return input_tensor

def add_shape_const_node(net_def, op, values, name):
  print ('Add const node: ', op.name + '/' + name)
  global node_count
  tensor = net_def.tensors.add()
  node_name = op.name + '/' + name
  tensor.name = node_name + ':0'
  tensor.node_id = node_count
  node_count += 1
  register_node_id(node_name, tensor.node_id)
  tensor.data_type =  mace_pb2.DT_INT32
  tensor.dims.extend(values)
  return tensor.name

def register_node_id(node_name, node_id):
  global node_ids
  node_ids[node_name] = node_id

def convert_ops(unresolved_ops, net_def, output_node, dsp_ops):
  global node_count

  ops_count = len(unresolved_ops)
  resolved_count = 1
  first_op = unresolved_ops[0]

  print ('Op: ', first_op.name, first_op.type, first_op.outputs[0].shape)

  if first_op.name in resolved_ops:
    pass

  elif first_op.type == 'Const':
    print ('Add const node: ', first_op.name)
    tf_tensor = first_op.outputs[0].eval()
    tensor = net_def.tensors.add()
    tensor.name = first_op.outputs[0].name
    tensor.node_id = node_count
    node_count += 1
    register_node_id(tensor.name.split(':')[0], tensor.node_id)
    tensor.data_type = find_dtype(first_op.outputs[0].dtype)
    shape = list(tf_tensor.shape)
    if len(shape) > 0:
      tensor.dims.extend(shape)
    if first_op.outputs[0].dtype == tf.float32:
      tensor.float_data.extend(tf_tensor.astype(float).flat)
    elif first_op.outputs[0].dtype == tf.int32 or \
            first_op.outputs[0].dtype == tf.int8 or \
            first_op.outputs[0].dtype == tf.int16 or \
            first_op.outputs[0].dtype == tf.quint8 or \
            first_op.outputs[0].dtype == tf.quint16:
      tensor.int32_data.extend(tf_tensor.astype(int).flat)

  else:
    op_def = net_def.op.add()
    op_def.name = first_op.name
    op_def.type = dsp_ops.map_nn_op(first_op.type)
    op_def.node_id = node_count
    node_count += 1
    op_def.padding = padding_mode['NA']

    if len(first_op.outputs) > 0 and first_op.type == 'Dequantize' \
        and len(first_op.outputs[0].consumers()) > 0 \
        and (first_op.outputs[0].consumers()[0].type == 'SpaceToBatchND' \
        or first_op.outputs[0].consumers()[0].type == 'BatchToSpaceND'):
      input_tensor = first_op.inputs[0]
      min_tensor = first_op.inputs[1]
      max_tensor = first_op.inputs[2]
      s2b_op = first_op.outputs[0].consumers()[0]
      reshape_op = s2b_op.outputs[0].consumers()[0]
      min_op = reshape_op.outputs[0].consumers()[0]
      max_op = reshape_op.outputs[0].consumers()[1]
      quantize_op = min_op.outputs[0].consumers()[0]
      resolved_ops.add(s2b_op.name)
      resolved_ops.add(reshape_op.name)
      resolved_ops.add(min_op.name)
      resolved_ops.add(max_op.name)
      resolved_ops.add(quantize_op.name)

      op_def.name = quantize_op.name
      op_def.type = dsp_ops.map_nn_op('Quantized' + s2b_op.type)
      op_def.input.append(input_tensor.name)
      op_def.input.extend([t.name for t in s2b_op.inputs[1:]])
      op_def.input.extend([min_tensor.name, max_tensor.name])
      op_def.out_max_byte_size.extend([max_elem_size(out) for out in quantize_op.outputs])

    elif has_padding_and_strides(first_op):
      op_def.padding = padding_mode[first_op.get_attr('padding')]
      op_def.input.extend([t.name for t in first_op.inputs])
      if 'ksize' in first_op.node_def.attr:
        ksize = first_op.get_attr('ksize')
        ksize_tensor = add_shape_const_node(net_def, first_op, ksize, 'ksize')
        op_def.input.extend([ksize_tensor])
      strides = first_op.get_attr('strides')
      strides_tensor = add_shape_const_node(net_def, first_op, strides, 'strides')
      op_def.input.extend([strides_tensor])
      op_def.out_max_byte_size.extend([max_elem_size(out) for out in first_op.outputs])
    elif is_node_flatten_reshape(first_op):
      op_def.type = 'Flatten'
      op_def.input.extend([t.name for t in first_op.inputs])
      op_def.out_max_byte_size.extend([max_elem_size(out) for out in first_op.outputs])
    elif dsp_ops.has_op(first_op.type):
      op_def.input.extend([t.name for t in first_op.inputs])
      op_def.out_max_byte_size.extend([max_elem_size(out) for out in first_op.outputs])

      if first_op.type == 'Placeholder':
        input_info = net_def.input_info.add()
        input_info.name = op_def.name
        input_info.node_id = op_def.node_id
        input_info.dims.extend(first_op.outputs[0].shape.as_list())
        input_info.max_byte_size = max_elem_size(first_op.outputs[0])
        input_info.data_type = find_dtype(first_op.outputs[0].dtype)
      elif first_op.name == output_node:
        output_info = net_def.output_info.add()
        output_info.name = op_def.name
        output_info.node_id = op_def.node_id
        output_info.dims.extend(first_op.outputs[0].shape.as_list())
        output_info.max_byte_size = max_elem_size(first_op.outputs[0])
        output_info.data_type = find_dtype(first_op.outputs[0].dtype)
    else:
      raise Exception('Unsupported op: ', first_op)

    register_node_id(op_def.name, op_def.node_id)

    print ('Add op node: ', first_op.name)
    for t in op_def.input:
      node, port = t.split(':')
      node_id = node_ids[node]
      node_input = op_def.node_input.add()
      node_input.node_id = node_id
      node_input.output_port = int(port)

    resolved_ops.add(first_op.name)

  for i in range(resolved_count):
    del unresolved_ops[0]

def add_output_node(net_def, output_node):
  global node_count
  op_def = net_def.op.add()
  op_def.name = 'output'
  op_def.type = 'OUTPUT'
  op_def.node_id = node_count
  node_count += 1
  register_node_id(op_def.name, op_def.node_id)
  op_def.input.extend([output_node + ':0'])
  node_input = op_def.node_input.add()
  node_input.node_id = node_ids[output_node]
  node_input.output_port = 0

def convert_to_mace_pb(input_graph_def, input_dim, output_node):
  """
    nnlib does not have batch norm, so use tensorflow optimizer to fold
     batch norm with convolution. The fold optimization reorders ops, so
     we sort ops first by topology.
  """
  input_graph_def = tf_graph_util.sort_graph(input_graph_def)
  inputs = input_dim.split(';')
  input_shape = {}
  for input in inputs:
    input_name_shape = input.split(',')
    name = input_name_shape[0]
    shape = [int(d) for d in input_name_shape[1:]]
    input_shape[name] = shape

  net_def = mace_pb2.NetDef()

  for node in input_graph_def.node:
    if node.op == 'Placeholder':
      node.attr['shape'].shape.unknown_rank = False
      for d in input_shape[node.name]:
        dim = node.attr['shape'].shape.dim.add()
        dim.size = d

  with tf.Session() as session:
    with session.graph.as_default() as graph:
      tf.import_graph_def(input_graph_def, name="")
      ops = graph.get_operations()
      dsp_ops = DspOps()
      # convert const node
      unresolved_ops = [op for op in ops if op.type == 'Const']
      while len(unresolved_ops) > 0:
        convert_ops(unresolved_ops, net_def, output_node, dsp_ops)

      # convert op node
      unresolved_ops = [op for op in ops if op.type != 'Const']
      while len(unresolved_ops) > 0:
        convert_ops(unresolved_ops, net_def, output_node, dsp_ops)

      add_output_node(net_def, output_node)

  return net_def
