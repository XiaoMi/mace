# Copyright 2018 Xiaomi, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import sys

import jinja2

# python encrypt_opencl_codegen.py --cl_kernel_dir=./mace/kernels/opencl/cl/  \
#     --output_path=./mace/codegen/opencl_encrypt/opencl_encrypted_program.cc

FLAGS = None

encrypt_lookup_table = "Mobile-AI-Compute-Engine"


def encrypt_code(code_str):
    encrypted_arr = []
    for i in range(len(code_str)):
        encrypted_char = hex(
            ord(code_str[i]) ^ ord(
                encrypt_lookup_table[i % len(encrypt_lookup_table)]))
        encrypted_arr.append(encrypted_char)
    return encrypted_arr


def encrypt_opencl_codegen(cl_kernel_dir, output_path):
    if not os.path.exists(cl_kernel_dir):
        print("Input cl_kernel_dir " + cl_kernel_dir + " doesn't exist!")

    header_code = ""
    for file_name in os.listdir(cl_kernel_dir):
        file_path = os.path.join(cl_kernel_dir, file_name)
        if file_path[-2:] == ".h":
            with open(file_path, "r") as f:
                header_code += f.read()

    encrypted_code_maps = {}
    for file_name in os.listdir(cl_kernel_dir):
        file_path = os.path.join(cl_kernel_dir, file_name)
        if file_path[-3:] == ".cl":
            with open(file_path, "r") as f:
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

    output_dir = os.path.dirname(output_path)
    if os.path.exists(output_dir):
        if os.path.isdir(output_dir):
            try:
                shutil.rmtree(output_dir)
            except OSError:
                raise RuntimeError(
                    "Cannot delete directory %s due to permission "
                    "error, inspect and remove manually" % output_dir)
        else:
            raise RuntimeError(
                "Cannot delete non-directory %s, inspect ",
                "and remove manually" % output_dir)
    os.makedirs(output_dir)

    with open(output_path, "w") as w_file:
        w_file.write(cpp_cl_encrypted_kernel)

    print('Generate OpenCL kernel done.')


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
    encrypt_opencl_codegen(FLAGS.cl_kernel_dir, FLAGS.output_path)
