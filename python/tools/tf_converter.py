import argparse
import sys
import hashlib
import tensorflow as tf
from tensorflow import gfile
from lib.proto import mace_pb2
from lib.python.tools import tf_converter_lib
from lib.python.tools import tf_dsp_converter_lib
from lib.python.tools import source_converter_lib

# ./bazel-bin/mace/python/tools/tf_converter --input quantized_test.pb --output quantized_test_dsp.pb --runtime dsp --input_dim input_node,1,28,28,3

FLAGS = None

def md5(fname):
  hash_md5 = hashlib.md5()
  with open(fname, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_md5.update(chunk)
  return hash_md5.hexdigest()

def main(unused_args):
  if not gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  mode_pb_checksum = md5(FLAGS.input)
  input_graph_def = tf.GraphDef()
  with gfile.Open(FLAGS.input, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

  if FLAGS.runtime == 'dsp':
    output_graph_def = tf_dsp_converter_lib.convert_to_mace_pb(
      input_graph_def, FLAGS.input_node, FLAGS.output_node)
  else:
    output_graph_def = tf_converter_lib.convert_to_mace_pb(
      input_graph_def, FLAGS.input_node, FLAGS.output_node, FLAGS.data_type, FLAGS.runtime, FLAGS.winograd)

  if FLAGS.output_type == 'source':
    source_converter_lib.convert_to_source(output_graph_def, mode_pb_checksum, FLAGS.template, FLAGS.obfuscate,
      FLAGS.model_tag, FLAGS.output, FLAGS.runtime)
  else:
    with gfile.GFile(FLAGS.output, "wb") as f:
      f.write(output_graph_def.SerializeToString())
    with gfile.GFile(FLAGS.output + '_txt', "wb") as f:
      # output_graph_def.ClearField('tensors')
      f.write(str(output_graph_def))
  print("Model conversion is completed.")

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

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
    "--data_type",
    type=str,
    default='DT_FLOAT',
    help="e.g., DT_HALF/DT_FLOAT")
  parser.add_argument(
    "--output_type",
    type=str,
    default="pb",
    help="output type: source/pb")
  parser.add_argument(
    "--template",
    type=str,
    default="",
    help="template path")
  parser.add_argument(
    "--obfuscate",
    type=str2bool,
    nargs='?',
    const=False,
    default=False,
    help="obfuscate model names")
  parser.add_argument(
    "--model_tag",
    type=str,
    default="",
    help="model tag for generated function and namespace")
  parser.add_argument(
    "--winograd",
    type=str2bool,
    nargs='?',
    const=False,
    default=False,
    help="obfuscate model names")
  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
