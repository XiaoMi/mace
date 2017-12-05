import argparse
import sys
import tensorflow as tf
import numpy as np

from tensorflow import gfile

# Validation Flow:
# 1. Generate input data
#    python validate_icnet.py --generate_data 1 \
#          --random_seed 1
# 2. Use mace_run to run icnet on phone.
# 3. adb pull the result.
# 4. Compare output data of mace and tf
#    python validate_icnet.py --model_file opt_icnet.pb \
#        --input_file input_file \
#        --mace_out_file icnet.out


def generate_data(shape):
  np.random.seed(FLAGS.random_seed)
  data = np.random.random(shape)
  print FLAGS.input_file
  data.astype(np.float32).tofile(FLAGS.input_file)
  print "Generate input file done."

def load_data(file):
  return np.fromfile(file=file, dtype=np.float32)

def valid_output(out_shape, mace_out_file, tf_out_value):
  mace_out_value = load_data(mace_out_file)
  mace_out_value = mace_out_value.reshape(out_shape)
  res = np.allclose(tf_out_value, mace_out_value, rtol=0, atol=1e-5)
  print 'Passed! Haha' if res else 'Failed! Oops'


def run_model(input_shape):
  if not gfile.Exists(FLAGS.model_file):
    print("Input graph file '" + FLAGS.model_file + "' does not exist!")
    return -1

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
        
        output_value = session.run(output_node, feed_dict={input_node: [input_value]})
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
    "--input_shape",
    type=str,
    default="512,512,3",
    help="input shape.")
  parser.add_argument(
    "--output_shape",
    type=str,
    default="1,512,512,2",
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
    help="Random seed for generate test case.")
  parser.add_argument(
    "--random_seed",
    type=int,
    default="0",
    help="Random seed for generate test case.")

  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)

