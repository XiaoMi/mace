import argparse
import os
import sys

import jinja2

# python encrypt_opencl_codegen.py --cl_kernel_dir=./mace/kernels/opencl/cl/  \
#     --output_path=./mace/codegen/opencl_encrypt/opencl_encrypted_program.cc

FLAGS = None

encrypt_lookup_table = "Xiaomi-AI-Platform-Mace"


def encrypt_code(code_str):
    encrypted_arr = []
    for i in range(len(code_str)):
        encrypted_char = hex(
            ord(code_str[i]) ^ ord(
                encrypt_lookup_table[i % len(encrypt_lookup_table)]))
        encrypted_arr.append(encrypted_char)
    return encrypted_arr


def main(unused_args):
    if not os.path.exists(FLAGS.cl_kernel_dir):
        print("Input cl_kernel_dir " + FLAGS.cl_kernel_dir + " doesn't exist!")

    header_code = ""
    for file_name in os.listdir(FLAGS.cl_kernel_dir):
        file_path = os.path.join(FLAGS.cl_kernel_dir, file_name)
        if file_path[-2:] == ".h":
            f = open(file_path, "r")
            header_code += f.read()

    encrypted_code_maps = {}
    for file_name in os.listdir(FLAGS.cl_kernel_dir):
        file_path = os.path.join(FLAGS.cl_kernel_dir, file_name)
        if file_path[-3:] == ".cl":
            f = open(file_path, "r")
            code_str = ""
            for line in f.readlines():
                if "#include <common.h>" in line:
                    code_str += header_code
                else:
                    code_str += line
            encrypted_code_arr = encrypt_code(code_str)
            encrypted_code_maps[file_name[:-3]] = encrypted_code_arr

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(sys.path[0]))
    cpp_cl_encrypted_kernel = env.get_template(
        'str2vec_maps.cc.jinja2').render(
            maps=encrypted_code_maps,
            data_type='unsigned char',
            variable_name='kEncryptedProgramMap')

    if os.path.isfile(FLAGS.output_path):
        os.remove(FLAGS.output_path)
    w_file = open(FLAGS.output_path, "w")
    w_file.write(cpp_cl_encrypted_kernel)
    w_file.close()

    print("Generate encrypted opencl source done!")


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cl_kernel_dir",
        type=str,
        default="./mace/kernels/opencl/cl/",
        help="The cl kernels directory.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./mace/examples/codegen/opencl/opencl_encrypted_program.cc",
        help="The path of encrypted opencl kernels.")
    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)
