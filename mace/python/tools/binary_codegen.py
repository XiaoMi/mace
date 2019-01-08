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
import sys
import struct

import jinja2

import numpy as np
import six


# python mace/python/tools/binary_codegen.py \
#     --binary_dirs=${BIN_FILE} \
#     --binary_file_name=mace_run.config \
#     --output_path=${CODE_GEN_PATH} --variable_name=kTuningParamsData

FLAGS = None


def generate_cpp_source(binary_dirs, binary_file_name, variable_name):
    data_map = {}
    for binary_dir in binary_dirs.split(","):
        binary_path = os.path.join(binary_dir, binary_file_name)
        if not os.path.exists(binary_path):
            continue

        with open(binary_path, "rb") as f:
            binary_array = np.fromfile(f, dtype=np.uint8)

        six.print_("Generate binary from", binary_path)
        idx = 0
        size, = struct.unpack("Q", binary_array[idx:idx + 8])
        idx += 8
        for _ in six.moves.range(size):
            key_size, = struct.unpack("i", binary_array[idx:idx + 4])
            idx += 4
            key, = struct.unpack(
                str(key_size) + "s", binary_array[idx:idx + key_size])
            idx += key_size
            params_size, = struct.unpack("i", binary_array[idx:idx + 4])
            idx += 4
            data_map[key] = []
            count = params_size / 4
            params = struct.unpack(
                str(count) + "i", binary_array[idx:idx + params_size])
            for i in params:
                data_map[key].append(i)
            idx += params_size

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(sys.path[0]))
    return env.get_template('str2vec_maps.cc.jinja2').render(
        maps=data_map,
        data_type='unsigned int',
        variable_name=variable_name)


def tuning_param_codegen(binary_dirs,
                         binary_file_name,
                         output_path,
                         variable_name):
    cpp_binary_source = generate_cpp_source(
            binary_dirs, binary_file_name, variable_name)
    if os.path.isfile(output_path):
        os.remove(output_path)
    with open(output_path, "w") as w_file:
        w_file.write(cpp_binary_source)


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary_dirs", type=str, default="", help="The binaries file path.")
    parser.add_argument(
        "--binary_file_name",
        type=str,
        default="mace_run.config",
        help="The binary file name.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="The path of generated C++ source file which contains the binary."
    )
    parser.add_argument(
        "--variable_name",
        type=str,
        default="kTuningParamsData",
        help="global variable name.")
    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    tuning_param_codegen(FLAGS.binary_dirs,
                         FLAGS.binary_file_name,
                         FLAGS.output_path,
                         FLAGS.variable_name)
