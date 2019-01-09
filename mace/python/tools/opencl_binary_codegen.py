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
import jinja2
import os
import sys

import numpy as np

FLAGS = None


def generate_opencl_code(binary_file_name, load_func_name, size_func_name,
                         output_path):
    binary_array = []
    if os.path.exists(binary_file_name):
        with open(binary_file_name, 'rb') as f:
            binary_array = np.fromfile(f, dtype=np.uint8)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(sys.path[0]))
    content = env.get_template('file_binary.cc.jinja2').render(
        data=binary_array,
        data_size=len(binary_array),
        load_func_name=load_func_name,
        size_func_name=size_func_name)

    if os.path.isfile(output_path):
        os.remove(output_path)
    with open(output_path, "w") as w_file:
        w_file.write(content)


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_name",
        type=str,
        default="opencl_binary.bin",
        help="The binary file name.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="The path of generated C++ source file which contains the binary."
    )
    parser.add_argument(
        "--load_func_name",
        type=str,
        default="LoadData",
        help="load interface name.")
    parser.add_argument(
        "--size_func_name",
        type=str,
        default="DataSize",
        help="size function name.")
    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    generate_opencl_code(FLAGS.file_name,
                         FLAGS.interface_name,
                         FLAGS.output_path)
