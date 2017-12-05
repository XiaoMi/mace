import argparse
import sys
import tensorflow as tf
from tensorflow import gfile
from mace.python.tools import tf_converter_lib
from mace.python.tools import tf_dsp_converter_lib

# ./bazel-bin/mace/python/tools/tf_converter --input quantized_test.pb --output quantized_test_dsp.pb --runtime dsp --input_dim input_node,1,28,28,3

FLAGS = None

def main(unused_args):
  if not gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  input_graph_def = tf.GraphDef()
  with gfile.Open(FLAGS.input, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

  if FLAGS.runtime == 'dsp':
    output_graph_def = tf_dsp_converter_lib.convert_to_mace_pb(
      input_graph_def, FLAGS.input_node, FLAGS.output_node, FLAGS.prequantize)
  else:
    output_graph_def = tf_converter_lib.convert_to_mace_pb(
      input_graph_def, FLAGS.runtime)

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
    help="Runtime: cpu/gpu/dsp.")
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
  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
