import argparse
import sys
import hashlib
import os.path
from lib.python.tools import source_converter_lib

# ./bazel-bin/mace/python/tools/tf_converter --model_file quantized_test.pb --output quantized_test_dsp.pb --runtime dsp --input_dim input_node,1,28,28,3

FLAGS = None

def file_checksum(fname):
  hash_func = hashlib.sha256()
  with open(fname, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_func.update(chunk)
  return hash_func.hexdigest()

def main(unused_args):
  if not os.path.isfile(FLAGS.model_file):
    print("Input graph file '" + FLAGS.model_file + "' does not exist!")
    sys.exit(-1)

  model_checksum = file_checksum(FLAGS.model_file)
  if FLAGS.model_checksum != "" and FLAGS.model_checksum != model_checksum:
    print("Model checksum mismatch: %s != %s" % (model_checksum, FLAGS.model_checksum))
    sys.exit(-1)

  if FLAGS.platform == 'caffe':
    if not os.path.isfile(FLAGS.weight_file):
      print("Input weight file '" + FLAGS.weight_file + "' does not exist!")
      sys.exit(-1)

    weight_checksum = file_checksum(FLAGS.weight_file)
    if FLAGS.weight_checksum != "" and FLAGS.weight_checksum != weight_checksum:
      print("Weight checksum mismatch: %s != %s" % (weight_checksum, FLAGS.weight_checksum))
      sys.exit(-1)

    if FLAGS.runtime == 'dsp':
      print("DSP not support caffe model yet.")
      sys.exit(-1)

    from lib.python.tools import caffe_converter_lib
    output_graph_def = caffe_converter_lib.convert_to_mace_pb(
      FLAGS.model_file, FLAGS.weight_file, FLAGS.input_node, FLAGS.input_shape, FLAGS.output_node,
      FLAGS.data_type, FLAGS.runtime, FLAGS.winograd)
  elif FLAGS.platform == 'tensorflow':
    if FLAGS.runtime == 'dsp':
      from lib.python.tools import tf_dsp_converter_lib
      output_graph_def = tf_dsp_converter_lib.convert_to_mace_pb(
        FLAGS.model_file, FLAGS.input_node, FLAGS.output_node, FLAGS.dsp_mode)
    else:
      from lib.python.tools import tf_converter_lib
      output_graph_def = tf_converter_lib.convert_to_mace_pb(
        FLAGS.model_file, FLAGS.input_node, FLAGS.input_shape, FLAGS.output_node,
        FLAGS.data_type, FLAGS.runtime, FLAGS.winograd)

  if FLAGS.output_type == 'source':
    source_converter_lib.convert_to_source(output_graph_def, model_checksum, FLAGS.template, FLAGS.obfuscate,
      FLAGS.model_tag, FLAGS.output, FLAGS.runtime, FLAGS.embed_model_data)
  else:
    with open(FLAGS.output, "wb") as f:
      f.write(output_graph_def.SerializeToString())
    with open(FLAGS.output + '_txt', "wb") as f:
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
    "--model_file",
    type=str,
    default="",
    help="TensorFlow \'GraphDef\' file to load, Caffe prototxt file to load.")
  parser.add_argument(
    "--weight_file",
    type=str,
    default="",
    help="Caffe data file to load.")
  parser.add_argument(
    "--model_checksum",
    type=str,
    default="",
    help="Model file sha256 checksum")
  parser.add_argument(
    "--weight_checksum",
    type=str,
    default="",
    help="Weight file sha256 checksum")
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
    help="open winograd convolution or not")
  parser.add_argument(
    "--dsp_mode",
    type=int,
    default=0,
    help="dsp run mode, defalut=0")
  parser.add_argument(
    "--input_shape",
    type=str,
    default="",
    help="input shape.")
  parser.add_argument(
    "--platform",
    type=str,
    default="tensorflow",
    help="tensorflow/caffe")
  parser.add_argument(
    "--embed_model_data",
    type=str2bool,
    default=True,
    help="input shape.")
  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
