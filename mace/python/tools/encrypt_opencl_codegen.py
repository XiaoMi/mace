# Copyright 2018 The MACE Authors. All Rights Reserved.
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

# python encrypt_opencl_codegen.py --cl_kernel_dir=./mace/ops/opencl/cl/  \
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


def create_output_dir(dir_path):
    if os.path.exists(dir_path):
        if os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
            except OSError:
                raise RuntimeError(
                    "Cannot delete directory %s due to permission "
                    "error, inspect and remove manually" % dir_path)
        else:
            raise RuntimeError(
                "Cannot delete non-directory %s, inspect ",
                "and remove manually" % dir_path)
    os.makedirs(dir_path)


def write_cl_encrypted_kernel_to_file(
        encrypted_code_maps, template_path, output_path):
    cwd = os.path.dirname(__file__)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(cwd))
    cl_encrypted_kernel = env.get_template(template_path).render(
        tag='codegen',
        maps=encrypted_code_maps,
        data_type='unsigned char',
        variable_name='kEncryptedProgramMap')
    with open(output_path, "w") as w_file:
        w_file.write(cl_encrypted_kernel)


def get_module_key(file_name):
    module_key = None
    if file_name[-3:] == ".cl":
        module_key = file_name[:-3]
    elif file_name[-2:] == ".h":
        module_key = file_name

    return module_key


def encrypt_opencl_codegen(cl_kernel_dir, output_path):
    if not os.path.exists(cl_kernel_dir):
        print("Input cl_kernel_dir " + cl_kernel_dir + " doesn't exist!")

    encrypted_code_maps = {}
    for file_name in os.listdir(cl_kernel_dir):
        file_path = os.path.join(cl_kernel_dir, file_name)
        module_key = get_module_key(file_name)
        if len(module_key) > 0:
            with open(file_path, "r") as f:
                code_str = ""
                headers = []
                for line in f.readlines():
                    if "#include <common.h>" in line:
                        headers.append(get_module_key("common.h"))
                    else:
                        code_str += line
                encrypted_code_arr = encrypt_code(code_str)
                encrypted_code = {}
                encrypted_code['headers'] = headers
                encrypted_code['code'] = encrypted_code_arr
                encrypted_code_maps[module_key] = encrypted_code

    create_output_dir(os.path.dirname(output_path))
    write_cl_encrypted_kernel_to_file(
        encrypted_code_maps, 'str2vec_maps.cc.jinja2', output_path)
    output_path_h = output_path.replace('.cc', '.h')
    write_cl_encrypted_kernel_to_file(
        encrypted_code_maps, 'str2vec_maps.h.jinja2', output_path_h)

    print('Generate OpenCL kernel done.')


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cl_kernel_dir",
        type=str,
        default="./mace/ops/opencl/cl/",
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
