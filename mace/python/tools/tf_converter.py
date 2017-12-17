import argparse
import sys
import tensorflow as tf
from tensorflow import gfile
from mace.proto import mace_pb2
from mace.python.tools import tf_converter_lib
from mace.python.tools import tf_dsp_converter_lib
import struct
from jinja2 import Environment, FileSystemLoader
import os

# ./bazel-bin/mace/python/tools/tf_converter --input quantized_test.pb --output quantized_test_dsp.pb --runtime dsp --input_dim input_node,1,28,28,3

FLAGS = None

class TensorInfo:
  def __init__(self, t):
    self.name = t.name
    if t.data_type == mace_pb2.DT_FLOAT:
      self.data = bytearray(struct.pack('%sf' % len(t.float_data), *t.float_data))
    elif t.data_type == mace_pb2.DT_INT32:
      self.data = bytearray(struct.pack('%si' % len(t.int32_data), *t.int32_data))

def stringfy(value):
  return ', '.join('"{0}"'.format(w) for w in value)

def convert_to_source(net_def):
  # Capture our current directory
  template_dir = os.path.dirname(FLAGS.template)
  template_name = os.path.basename(FLAGS.template)
  print template_dir

  # Create the jinja2 environment.
  # Notice the use of trim_blocks, which greatly helps control whitespace.
  j2_env = Environment(loader=FileSystemLoader(template_dir),
    trim_blocks=True)
  j2_env.filters['stringfy'] = stringfy
  counter = 0
  output_dir = os.path.dirname(FLAGS.output) + '/'
  for t in net_def.tensors:
    source = j2_env.get_template(template_name).render(
      tensor = TensorInfo(t),
      mode = 0,
    )
    with gfile.GFile(output_dir + str(counter) + '.cc', "wb") as f:
      f.write(source)
    counter += 1


  tensors = [TensorInfo(t) for t in net_def.tensors]
  source = j2_env.get_template(template_name).render(
    tensors = tensors,
    net = net_def,
    mode = 1
  )
  with gfile.GFile(FLAGS.output, "wb") as f:
    f.write(source)

def main(unused_args):
  if not gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  input_graph_def = tf.GraphDef()
  with gfile.Open(FLAGS.input, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

  print 'done'
  if FLAGS.runtime == 'dsp':
    output_graph_def = tf_dsp_converter_lib.convert_to_mace_pb(
      input_graph_def, FLAGS.input_node, FLAGS.output_node, FLAGS.prequantize)
  else:
    output_graph_def = tf_converter_lib.convert_to_mace_pb(
      input_graph_def, FLAGS.input_node, FLAGS.output_node, FLAGS.data_type, FLAGS.runtime)

  if FLAGS.output_type == 'source':
    convert_to_source(output_graph_def)
  else:
    with gfile.GFile(FLAGS.output, "wb") as f:
      f.write(output_graph_def.SerializeToString())
    with gfile.GFile(FLAGS.output + '_txt', "wb") as f:
      # output_graph_def.ClearField('tensors')
      f.write(str(output_graph_def))


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
    "--input",
    type=str,
    default="",
    help="TensorFlow \'GraphDef\' file to load.")
  parser.add_argument(
    "--output",
    type=str,
    default="",
    help="File to save the output graph to.")
  parser.add_argument(
    "--runtime",
    type=str,
    default="cpu",
    help="Runtime: cpu/gpu/dsp")
  parser.add_argument(
    "--input_node",
    type=str,
    default="input_node",
    help="e.g., input_node")
  parser.add_argument(
    "--output_node",
    type=str,
    default="softmax",
    help="e.g., softmax")
  parser.add_argument(
    "--prequantize",
    type=bool,
    default=False,
    help="e.g., False")
  parser.add_argument(
    "--data_type",
    type=str,
    default='DT_FLOAT',
    help="e.g., DT_HALF/DT_FLOAT")
  parser.add_argument(
    "--output_type",
    type=str,
    default="source",
    help="output type: source/pb")
  parser.add_argument(
    "--template",
    type=str,
    default="",
    help="template path")
  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
