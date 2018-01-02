import argparse
import os
import sys

import jinja2

# python mace/python/tools/read_tuning_codegen.py \
#   --output_path=./mace/codegen/tuning/read_tuning_params.cc

FLAGS = None

def main(unused_args):
  env = jinja2.Environment(loader=jinja2.FileSystemLoader(sys.path[0]))
  cpp_cl_encrypted_kernel = env.get_template('embed_code.cc.tmpl').render(
      data_type='unsigned int',
      mode='read_tuning_config')

  if os.path.isfile(FLAGS.output_path):
    os.remove(FLAGS.output_path)
  w_file = open(FLAGS.output_path, "w")
  w_file.write(cpp_cl_encrypted_kernel)
  w_file.close()

  print("Generate reading tuning method done!")


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_path",
      type=str,
      default="./mace/examples/codegen/tuning/read_tuning_params.cc",
      help="The path of codes to read tuning params.")
  return parser.parse_known_args()


if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  main(unused_args=[sys.argv[0]] + unparsed)
