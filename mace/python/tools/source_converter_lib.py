import struct
import os
import uuid
import numpy as np

from tensorflow import gfile
from mace.proto import mace_pb2
from jinja2 import Environment, FileSystemLoader


GENERATED_NAME = set()

def generate_random_name():
  name = '_' + uuid.uuid4().hex[:7].upper()
  while name in GENERATED_NAME:
    name = '_' + uuid.uuid4().hex[:7].upper()
  GENERATED_NAME.add(name)
  return name

def generate_tensor_map(tensors):
  tensor_map = {}
  for t in tensors:
    if not tensor_map.has_key(t.name):
      tensor_map[t.name] = generate_random_name()
  return tensor_map

def generate_in_out_map(ops, tensor_map):
  in_out_map = {}
  for op in ops:
    op.name = generate_random_name()
    for input_name in op.input:
        if not in_out_map.has_key(input_name):
          if tensor_map.has_key(input_name):
            in_out_map[input_name] = tensor_map[input_name]
          else:
            in_out_map[input_name] = generate_random_name()
    for output_name in op.output:
      if not in_out_map.has_key(output_name):
        if tensor_map.has_key(output_name):
          in_out_map[output_name] = tensor_map[output_name]
        else:
          in_out_map[output_name] = generate_random_name()
  return in_out_map

def confuse_name(net_def):
  input_node = "mace_input_node"
  output_node = "mace_output_node"
  tensor_map = generate_tensor_map(net_def.tensors)
  in_out_map = generate_in_out_map(net_def.op, tensor_map)
  for t in net_def.tensors:
    if input_node not in t.name and output_node not in t.name:
      t.name = tensor_map[t.name]
  for op in net_def.op:
    for i in range(len(op.input)):
      if input_node not in op.input[i]:
        op.input[i] = in_out_map[op.input[i]]
    for i in range(len(op.output)):
      if output_node not in op.output[i]:
        op.output[i] = in_out_map[op.output[i]]

def rename_tensor(net_def):
  tensor_map = {}
  for t in net_def.tensors:
    if not tensor_map.has_key(t.name):
      tensor_map[t.name] = "_" + t.name[:-2].replace("/", "_")
      t.name = tensor_map[t.name]
  for op in net_def.op:
    for i in range(len(op.input)):
      if tensor_map.has_key(op.input[i]):
        op.input[i] = tensor_map[op.input[i]]
    for i in range(len(op.output)):
      if tensor_map.has_key(op.output[i]):
        op.output[i] = tensor_map[op.output[i]]

class TensorInfo:
  def __init__(self, t):
    self.name = t.name
    self.data_type = mace_pb2.DataType.Name(t.data_type)
    if t.data_type == mace_pb2.DT_FLOAT:
      self.data = bytearray(struct.pack('%sf' % len(t.float_data), *t.float_data))
    elif t.data_type == mace_pb2.DT_INT32:
      self.data = bytearray(struct.pack('%si' % len(t.int32_data), *t.int32_data))
    elif t.data_type == mace_pb2.DT_UINT8:
      self.data = bytearray(np.array(t.int32_data).astype(np.uint8).tolist())

def stringfy(value):
  return ', '.join('"{0}"'.format(w) for w in value)

def convert_to_source(net_def, template, confuse, model_tag, output, runtime):
  if confuse:
    confuse_name(net_def)
  else:
    rename_tensor(net_def)

  # Capture our current directory
  template_dir = os.path.dirname(template)
  template_name = os.path.basename(template)
  print template_dir

  # Create the jinja2 environment.
  j2_env = Environment(loader=FileSystemLoader(template_dir),
    trim_blocks=True)
  j2_env.filters['stringfy'] = stringfy
  counter = 0
  output_dir = os.path.dirname(output) + '/'
  # generate tensor source files
  for t in net_def.tensors:
    source = j2_env.get_template(template_name).render(
      tensor_info = TensorInfo(t),
      tensor = t,
      tag = model_tag,
      mode = 0,
      runtime = runtime,
    )
    with gfile.GFile(output_dir + 'tensor' + str(counter) + '.cc', "wb") as f:
      f.write(source)
    counter += 1

  # generate op source files
  counter = 0
  op_size = len(net_def.op)
  for start in range(0, op_size, 10):
    source = j2_env.get_template(template_name).render(
      start = start,
      end = min(start+10, op_size),
      net = net_def,
      tag = model_tag,
      mode = 1,
      runtime = runtime,
    )
    with gfile.GFile(output_dir + 'op' + str(counter) + '.cc', "wb") as f:
      f.write(source)
    counter += 1

  # generate model source files
  tensors = [TensorInfo(t) for t in net_def.tensors]
  source = j2_env.get_template(template_name).render(
    tensors = tensors,
    net = net_def,
    tag = model_tag,
    mode = 2,
    runtime = runtime,
  )
  with gfile.GFile(output, "wb") as f:
    f.write(source)
