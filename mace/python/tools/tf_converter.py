import argparse
import sys
import tensorflow as tf
from tensorflow import gfile
from mace.python.tools import tf_converter_lib

FLAGS = None


def main(unused_args):
  if not gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  input_graph_def = tf.GraphDef()
  with gfile.Open(FLAGS.input, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

  output_graph_def = tf_converter_lib.convert_to_mace_pb(
    input_graph_def)

  with gfile.GFile(FLAGS.output, "wb") as f:
    f.write(output_graph_def.SerializeToString())
  with gfile.GFile(FLAGS.output + '_txt', "wb") as f:
    output_graph_def.ClearField('tensors')
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
  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
