import argparse
import os
import sys

import numpy as np

import jinja2

# python mace/python/tools/opencl_codegen.py \
#     --cl_binary_dir=${CL_BIN_DIR} --output_path=${CL_HEADER_PATH}

FLAGS = None


def generate_cpp_source():
  maps = {}
  for file_name in os.listdir(FLAGS.cl_binary_dir):
    file_path = os.path.join(FLAGS.cl_binary_dir, file_name)
    if file_path[-4:] == ".bin":
      # read binary
      f = open(file_path, "rb")
      binary_array = np.fromfile(f, dtype=np.uint8)
      f.close()

      maps[file_name[:-4]] = []
      for ele in binary_array:
        maps[file_name[:-4]].append(hex(ele))

  env = jinja2.Environment(loader=jinja2.FileSystemLoader(sys.path[0]))
  return env.get_template('embed_code.cc.tmpl').render(
    maps = maps,
    data_type = 'unsigned char',
    variable_name = 'kCompiledProgramMap',
    mode="cl_binary"
  )


def main(unused_args):
  if not os.path.exists(FLAGS.cl_binary_dir):
    print("Input cl_binary_dir " + FLAGS.cl_binary_dir + " doesn't exist!")

  cpp_cl_binary_source = generate_cpp_source()
  if os.path.isfile(FLAGS.output_path):
    os.remove(FLAGS.output_path)
  w_file = open(FLAGS.output_path, "w")
  w_file.write(cpp_cl_binary_source)
  w_file.close()


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--cl_binary_dir",
      type=str,
      default="./cl_bin/",
      help="The cl binaries directory.")
  parser.add_argument(
      "--output_path",
      type=str,
      default="./mace/examples/codegen/opencl/opencl_compiled_program.cc",
      help="The path of generated C++ header file which contains cl binaries.")
  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
