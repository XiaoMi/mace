import argparse
import os
import sys
import struct

import numpy as np

import jinja2

# python mace/python/tools/opencl_codegen.py \
#     --cl_binary_dirs=${CL_BIN_DIR} --output_path=${CL_HEADER_PATH}

FLAGS = None


def generate_cpp_source():
  cl_built_kernel_file_name = 'mace_cl_compiled_program.bin'
  cl_platform_info_file_name = 'mace_cl_platform_info.txt'
  maps = {}
  platform_info = ''
  for binary_dir in FLAGS.cl_binary_dirs.split(","):
    binary_path = os.path.join(binary_dir, cl_built_kernel_file_name)
    if not os.path.exists(binary_path):
      continue

    with open(binary_path, "rb") as f:
      binary_array = np.fromfile(f, dtype=np.uint8)

    idx = 0
    size, = struct.unpack("Q", binary_array[idx:idx+8])
    idx += 8
    for _ in xrange(size):
      key_size, = struct.unpack("i", binary_array[idx:idx+4])
      idx += 4
      key, = struct.unpack(str(key_size) + "s", binary_array[idx:idx+key_size])
      idx += key_size
      value_size, = struct.unpack("i", binary_array[idx:idx+4])
      idx += 4
      maps[key] = []
      value = struct.unpack(str(value_size) + "B",
                            binary_array[idx:idx+value_size])
      idx += value_size
      for ele in value:
        maps[key].append(hex(ele))

    cl_platform_info_path = os.path.join(binary_dir, cl_platform_info_file_name)
    with open(cl_platform_info_path, 'r') as f:
      curr_platform_info = f.read()
    if platform_info != "":
      assert(curr_platform_info == platform_info)
    platform_info = curr_platform_info

  env = jinja2.Environment(loader=jinja2.FileSystemLoader(sys.path[0]))
  return env.get_template('opencl_compiled_kernel.cc.jinja2').render(
    maps = maps,
    data_type = 'unsigned char',
    variable_name = 'kCompiledProgramMap',
    platform_info = platform_info,
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
      default="",
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
