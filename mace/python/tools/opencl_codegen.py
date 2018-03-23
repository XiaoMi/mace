import argparse
import os
import sys

import numpy as np

import jinja2

# python mace/python/tools/opencl_codegen.py \
#     --cl_binary_dirs=${CL_BIN_DIR} --output_path=${CL_HEADER_PATH}

FLAGS = None


def generate_cpp_source():
  maps = {}
  cl_binary_dir_arr = FLAGS.cl_binary_dirs.split(",")
  for cl_binary_dir in cl_binary_dir_arr:
    if not os.path.exists(cl_binary_dir):
      print("Input cl_binary_dir " + cl_binary_dir + " doesn't exist!")
    for file_name in os.listdir(cl_binary_dir):
      file_path = os.path.join(cl_binary_dir, file_name)
      if file_path[-4:] == ".bin":
        # read binary
        f = open(file_path, "rb")
        binary_array = np.fromfile(f, dtype=np.uint8)
        f.close()

        maps[file_name[:-4]] = []
        for ele in binary_array:
          maps[file_name[:-4]].append(hex(ele))

  env = jinja2.Environment(loader=jinja2.FileSystemLoader(sys.path[0]))
  return env.get_template('str2vec_maps.cc.jinja2').render(
    maps = maps,
    data_type = 'unsigned char',
    variable_name = 'kCompiledProgramMap'
  )


def main(unused_args):

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
      "--cl_binary_dirs",
      type=str,
      default="cl_bin0/,cl_bin1/,cl_bin2/",
      help="The cl binaries directories.")
  parser.add_argument(
      "--output_path",
      type=str,
      default="./mace/examples/codegen/opencl/opencl_compiled_program.cc",
      help="The path of generated C++ header file which contains cl binaries.")
  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
