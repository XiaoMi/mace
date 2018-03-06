import argparse
import sys
import os
import os.path
import numpy as np
import re
from scipy import spatial

# Validation Flow:
# 1. Generate input data
#    python generate_data.py \
#        --input_node input_node \
#        --input_shape 1,64,64,3 \
#        --input_file input_file
#

def generate_data(name, shape):
  np.random.seed()
  data = np.random.random(shape) * 2 - 1
  input_file_name = FLAGS.input_file + "_" + re.sub('[^0-9a-zA-Z]+', '_', name)
  print 'Generate input file: ', input_file_name
  data.astype(np.float32).tofile(input_file_name)

def main(unused_args):
  input_names = [name for name in FLAGS.input_node.split(',')]
  input_shapes = [shape for shape in FLAGS.input_shape.split(':')]
  assert len(input_names) == len(input_shapes)
  for i in range(len(input_names)):
    shape = [int(x) for x in input_shapes[i].split(',')]
    generate_data(input_names[i], shape)
  print "Generate input file done."

def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
    "--input_file",
    type=str,
    default="",
    help="input file.")
  parser.add_argument(
    "--input_node",
    type=str,
    default="input_node",
    help="input node")
  parser.add_argument(
    "--input_shape",
    type=str,
    default="1,64,64,3",
    help="input shape.")

  return parser.parse_known_args()

if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)

