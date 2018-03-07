import argparse
import sys
import os
import os.path
import numpy as np
import re
from scipy import spatial

# Validation Flow:
# 1. Generate input data
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

def load_data(file):
  if os.path.isfile(file):
    return np.fromfile(file=file, dtype=np.float32)
  else:
    return np.empty([0])

def format_output_name(name):
  return re.sub('[^0-9a-zA-Z]+', '_', name)

def compare_output(output_name, mace_out_value, out_value):
  if mace_out_value.size != 0:
    similarity = (1 - spatial.distance.cosine(out_value.flat, mace_out_value.flat))
    print output_name, 'MACE VS', FLAGS.platform.upper(), 'similarity: ', similarity
    if (FLAGS.mace_runtime == "cpu" and similarity > 0.999) or \
        (FLAGS.mace_runtime == "gpu" and similarity > 0.995) or \
        (FLAGS.mace_runtime == "dsp" and similarity > 0.930):
      print '=======================Similarity Test Passed======================'
    else:
      print '=======================Similarity Test Failed======================'
  else:
    print '=======================Skip empty node==================='


def validate_tf_model(input_names, input_shapes, output_names):
  import tensorflow as tf
  if not os.path.isfile(FLAGS.model_file):
    print("Input graph file '" + FLAGS.model_file + "' does not exist!")
    sys.exit(-1)

  input_graph_def = tf.GraphDef()
  with open(FLAGS.model_file, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)
    tf.import_graph_def(input_graph_def, name="")

    with tf.Session() as session:
      with session.graph.as_default() as graph:
        tf.import_graph_def(input_graph_def, name="")
        input_dict = {}
        for i in range(len(input_names)):
          input_value = load_data(FLAGS.input_file + "_" + input_names[i])
          input_value = input_value.reshape(input_shapes[i])
          input_node = graph.get_tensor_by_name(input_names[i] + ':0')
          input_dict[input_node] = input_value

        output_nodes = []
        for name in output_names:
          output_nodes.extend([graph.get_tensor_by_name(name + ':0')])
        output_values = session.run(output_nodes, feed_dict=input_dict)
        for i in range(len(output_names)):
          output_file_name = FLAGS.mace_out_file + "_" + format_output_name(output_names[i])
          mace_out_value = load_data(output_file_name)
          compare_output(output_names[i], mace_out_value, output_values[i])

def validate_caffe_model(input_names, input_shapes, output_names, output_shapes):
  os.environ['GLOG_minloglevel'] = '1' # suprress Caffe verbose prints
  import caffe
  if not os.path.isfile(FLAGS.model_file):
    print("Input graph file '" + FLAGS.model_file + "' does not exist!")
    sys.exit(-1)
  if not os.path.isfile(FLAGS.weight_file):
    print("Input weight file '" + FLAGS.weight_file + "' does not exist!")
    sys.exit(-1)

  caffe.set_mode_cpu()

  net = caffe.Net(FLAGS.model_file, caffe.TEST, weights=FLAGS.weight_file)

  for i in range(len(input_names)):
    input_value = load_data(FLAGS.input_file + "_" + input_names[i])
    input_value = input_value.reshape(input_shapes[i]).transpose((0, 3, 1, 2))
    net.blobs[input_names[i]].data[0] = input_value

  net.forward()

  for i in range(len(output_names)):
    value = net.blobs[output_names[i]].data[0]
    out_shape = output_shapes[i]
    out_shape[1], out_shape[2], out_shape[3] = out_shape[3], out_shape[1], out_shape[2]
    value = value.reshape(out_shape).transpose((0, 2, 3, 1))
    output_file_name = FLAGS.mace_out_file + "_" + format_output_name(output_names[i])
    mace_out_value = load_data(output_file_name)
    compare_output(output_names[i], mace_out_value, value)

def main(unused_args):
  input_names = [name for name in FLAGS.input_node.split(',')]
  input_shape_strs = [shape for shape in FLAGS.input_shape.split(':')]
  input_shapes = [[int(x) for x in shape.split(',')] for shape in input_shape_strs]
  output_names = [name for name in FLAGS.output_node.split(',')]
  assert len(input_names) == len(input_shapes)

  if FLAGS.platform == 'tensorflow':
    validate_tf_model(input_names, input_shapes, output_names)
  elif FLAGS.platform == 'caffe':
    output_shape_strs = [shape for shape in FLAGS.output_shape.split(':')]
    output_shapes = [[int(x) for x in shape.split(',')] for shape in output_shape_strs]
    validate_caffe_model(input_names, input_shapes, output_names, output_shapes)

def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
    "--platform",
    type=str,
    default="",
    help="Tensorflow or Caffe.")
  parser.add_argument(
    "--model_file",
    type=str,
    default="",
    help="TensorFlow or Caffe \'GraphDef\' file to load.")
  parser.add_argument(
    "--weight_file",
    type=str,
    default="",
    help="caffe model file to load.")
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

  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)

