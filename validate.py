import argparse
import sys
import os
import os.path
import tensorflow as tf
import numpy as np
from scipy import spatial

from tensorflow import gfile

# Validation Flow:
# 1. Generate input data
#    python validate.py --generate_data true \
#        --input_file input_file
#        --input_shape 1,64,64,3
#
# 2. Use mace_run to run model on phone.
# 3. adb pull the result.
# 4. Compare output data of mace and tf
#    python validate.py --model_file tf_model_opt.pb \
#        --input_file input_file \
#        --mace_out_file output_file \
#        --input_node input_node \
#        --output_node output_node \
#        --input_shape 1,64,64,3 \
#        --output_shape 1,64,64,2

def generate_data(shape):
  np.random.seed()
  data = np.random.random(shape) * 2 - 1
  print FLAGS.input_file
  data.astype(np.float32).tofile(FLAGS.input_file)
  print "Generate input file done."

def load_data(file):
  if os.path.isfile(file):
    return np.fromfile(file=file, dtype=np.float32)
  else:
    return np.empty([0])

def valid_output(out_shape, mace_out_file, tf_out_value):
  mace_out_value = load_data(mace_out_file)
  if mace_out_value.size != 0:
    similarity = (1 - spatial.distance.cosine(tf_out_value.flat, mace_out_value))
    print 'MACE VS TF similarity: ', similarity
    if (FLAGS.mace_runtime == "cpu" and similarity > 0.999) or \
        (FLAGS.mace_runtime == "gpu" and similarity > 0.995) or \
        (FLAGS.mace_runtime == "dsp" and similarity > 0.930):
      print '=======================Similarity Test Passed======================'
    else:
      print '=======================Similarity Test Failed======================'
  else:
    print '=======================Skip empty node==================='


def run_model(input_shape):
  if not gfile.Exists(FLAGS.model_file):
    print("Input graph file '" + FLAGS.model_file + "' does not exist!")
    sys.exit(-1)

  input_graph_def = tf.GraphDef()
  with gfile.Open(FLAGS.model_file, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)
    tf.import_graph_def(input_graph_def, name="")

    with tf.Session() as session:
      with session.graph.as_default() as graph:
        tf.import_graph_def(input_graph_def, name="")
        input_node = graph.get_tensor_by_name(FLAGS.input_node + ':0')
        output_node = graph.get_tensor_by_name(FLAGS.output_node + ':0')

        input_value = load_data(FLAGS.input_file)
        input_value = input_value.reshape(input_shape)
        
        output_value = session.run(output_node, feed_dict={input_node: input_value})
        output_value.astype(np.float32).tofile( os.path.dirname(FLAGS.input_file) + '/tf_out')
        return output_value

def main(unused_args):
  input_shape = [int(x) for x in FLAGS.input_shape.split(',')]
  output_shape = [int(x) for x in FLAGS.output_shape.split(',')]
  if FLAGS.generate_data:
    generate_data(input_shape)
  else:
    output_value = run_model(input_shape)
    valid_output(output_shape, FLAGS.mace_out_file, output_value)


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
    "--model_file",
    type=str,
    default="",
    help="TensorFlow \'GraphDef\' file to load.")
  parser.add_argument(
    "--input_file",
    type=str,
    default="",
    help="input file.")
  parser.add_argument(
    "--mace_out_file",
    type=str,
    default="",
    help="mace output file to load.")
  parser.add_argument(
    "--mace_runtime",
    type=str,
    default="gpu",
    help="mace runtime device.")
  parser.add_argument(
    "--input_shape",
    type=str,
    default="1,64,64,3",
    help="input shape.")
  parser.add_argument(
    "--output_shape",
    type=str,
    default="1,64,64,2",
    help="output shape.")
  parser.add_argument(
    "--input_node",
    type=str,
    default="input_node",
    help="input node")
  parser.add_argument(
    "--output_node",
    type=str,
    default="output_node",
    help="output node")
  parser.add_argument(
    "--generate_data",
    type='bool',
    default="false",
    help="Generate data or not.")

  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)

