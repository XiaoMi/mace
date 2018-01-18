import argparse
import os
import sys
import struct

import jinja2

import numpy as np

# python mace/python/tools/binary_codegen.py \
#     --binary_file=${BIN_FILE} --output_path=${CODE_GEN_PATH} --variable_name=kTuningParamsData

FLAGS = None


def generate_cpp_source():
  data_map = {}
  if not os.path.exists(FLAGS.binary_file):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(sys.path[0]))
    return env.get_template('str2vec_maps.cc.tmpl').render(
      maps=data_map,
      data_type='unsigned int',
      variable_name=FLAGS.variable_name
    )

  with open(FLAGS.binary_file, "rb") as binary_file:
    binary_array = np.fromfile(binary_file, dtype=np.uint8)

  idx = 0
  size, = struct.unpack("Q", binary_array[idx:idx+8])
  print size
  idx += 8
  for _ in xrange(size):
    key_size, = struct.unpack("i", binary_array[idx:idx+4])
    idx += 4
    key, = struct.unpack(str(key_size) + "s", binary_array[idx:idx+key_size])
    idx += key_size
    params_size, = struct.unpack("i", binary_array[idx:idx+4])
    idx += 4
    data_map[key] = []
    count = params_size / 4
    params = struct.unpack(str(count) + "i", binary_array[idx:idx+params_size])
    for i in params:
      data_map[key].append(i)
    idx += params_size

  env = jinja2.Environment(loader=jinja2.FileSystemLoader(sys.path[0]))
  return env.get_template('str2vec_maps.cc.tmpl').render(
    maps = data_map,
    data_type = 'unsigned int',
    variable_name = FLAGS.variable_name
  )

def main(unused_args):
  cpp_binary_source = generate_cpp_source()
  if os.path.isfile(FLAGS.output_path):
    os.remove(FLAGS.output_path)
  w_file = open(FLAGS.output_path, "w")
  w_file.write(cpp_binary_source)
  w_file.close()

def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--binary_file",
      type=str,
      default="",
      help="The binaries file path.")
  parser.add_argument(
      "--output_path",
      type=str,
      default="",
      help="The path of generated C++ source file which contains the binary.")
  parser.add_argument(
    "--variable_name",
    type=str,
    default="kTuningParamsData",
    help="global variable name.")
  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
