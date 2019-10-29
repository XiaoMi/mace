# Copyright 2019 The MACE Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import jinja2
import os
import struct
import numpy as np

from utils import util
from utils.util import MaceLogger
from utils.util import mace_check


def generate_opencl_code(binary_file_name, load_func_name, size_func_name,
                         output_path):
    binary_array = []
    if os.path.exists(binary_file_name):
        with open(binary_file_name, 'rb') as f:
            binary_array = np.fromfile(f, dtype=np.uint8)

    cwd = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(cwd + "/template"))
    content = env.get_template('file_binary.cc.jinja2').render(
        data=binary_array,
        data_size=len(binary_array),
        load_func_name=load_func_name,
        size_func_name=size_func_name)

    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, "w") as w_file:
        w_file.write(content)


def merge_opencl_binaries(opencl_binaries,
                          output_file):
    platform_info_key = 'mace_opencl_precompiled_platform_info_key'

    kvs = {}
    for binary in opencl_binaries:
        if not os.path.exists(binary):
            MaceLogger.warning("OpenCL bin %s not found" % binary)
            continue

        with open(binary, "rb") as f:
            binary_array = np.fromfile(f, dtype=np.uint8)

        idx = 0
        size, = struct.unpack("Q", binary_array[idx:idx + 8])
        idx += 8
        for _ in range(size):
            key_size, = struct.unpack("i", binary_array[idx:idx + 4])
            idx += 4
            key, = struct.unpack(
                str(key_size) + "s", binary_array[idx:idx + key_size])
            idx += key_size
            value_size, = struct.unpack("i", binary_array[idx:idx + 4])
            idx += 4
            if key == platform_info_key and key in kvs:
                mace_check(
                    (kvs[key] == binary_array[idx:idx + value_size]).all(),
                    "There exists more than one OpenCL version for models:"
                    " %s vs %s " %
                    (kvs[key], binary_array[idx:idx + value_size]))
            else:
                kvs[key] = binary_array[idx:idx + value_size]
            idx += value_size

    output_byte_array = bytearray()
    data_size = len(kvs)
    output_byte_array.extend(struct.pack("Q", data_size))
    for key, value in kvs.items():
        key_size = len(key)
        output_byte_array.extend(struct.pack("i", key_size))
        output_byte_array.extend(struct.pack(str(key_size) + "s", key))
        value_size = len(value)
        output_byte_array.extend(struct.pack("i", value_size))
        output_byte_array.extend(value)

    np.array(output_byte_array).tofile(output_file)


def merge_opencl_parameters(params_files,
                            output_file):
    kvs = {}
    for params in params_files:
        if not os.path.exists(params):
            MaceLogger.warning("Tune param %s not found" % params)
            continue

        with open(params, "rb") as f:
            binary_array = np.fromfile(f, dtype=np.uint8)

        idx = 0
        size, = struct.unpack("Q", binary_array[idx:idx + 8])
        idx += 8
        for _ in range(size):
            key_size, = struct.unpack("i", binary_array[idx:idx + 4])
            idx += 4
            key, = struct.unpack(
                str(key_size) + "s", binary_array[idx:idx + key_size])
            idx += key_size
            value_size, = struct.unpack("i", binary_array[idx:idx + 4])
            idx += 4
            kvs[key] = binary_array[idx:idx + value_size]
            idx += value_size

    output_byte_array = bytearray()
    data_size = len(kvs)
    output_byte_array.extend(struct.pack("Q", data_size))
    for key, value in kvs.items():
        key_size = len(key)
        output_byte_array.extend(struct.pack("i", key_size))
        output_byte_array.extend(struct.pack(str(key_size) + "s", key))
        value_size = len(value)
        output_byte_array.extend(struct.pack("i", value_size))
        output_byte_array.extend(value)

    np.array(output_byte_array).tofile(output_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--binary_files',
        type=str,
        default="",
        help="opencl binary files")
    parser.add_argument(
        '--tuning_files',
        type=str,
        default="",
        help="tuning params files")
    parser.add_argument(
        '--output',
        type=str,
        default="build",
        help="output dir")
    parser.add_argument(
        "--gencode",
        action="store_true",
        help="generate code")
    flgs, _ = parser.parse_known_args()
    return flgs


if __name__ == '__main__':
    flags = parse_args()
    util.mkdir_p(flags.output)
    opencl_binary_files = []
    if flags.binary_files:
        opencl_binary_files = flags.binary_files.split(",")
    opencl_tuning_files = []
    if flags.tuning_files:
        opencl_tuning_files = flags.tuning_files.split(",")

    compiled_opencl_kernel_prefix = "compiled_opencl_kernel"
    tuned_opencl_parameter_prefix = "tuned_opencl_parameter"

    if not opencl_binary_files and not opencl_tuning_files:
        for root, dirs, files in os.walk("build", topdown=False):
            for name in files:
                if compiled_opencl_kernel_prefix in name:
                    opencl_binary_files.append(os.path.join(root, name))
                elif tuned_opencl_parameter_prefix in name:
                    opencl_tuning_files.append(os.path.join(root, name))

    opencl_dir = flags.output + "/opencl"
    util.mkdir_p(opencl_dir)
    merged_opencl_bin_file = "%s/%s.bin" % (opencl_dir,
                                            compiled_opencl_kernel_prefix)
    merged_opencl_tuning_file = "%s/%s.bin" % (opencl_dir,
                                               tuned_opencl_parameter_prefix)

    merge_opencl_binaries(opencl_binary_files,
                          merged_opencl_bin_file)
    if flags.gencode:
        util.mkdir_p('mace/codegen/opencl')
        generate_opencl_code(merged_opencl_bin_file,
                             "LoadOpenCLBinary",
                             "OpenCLBinarySize",
                             "mace/codegen/opencl/opencl_binary.cc")

    merge_opencl_binaries(opencl_tuning_files,
                          merged_opencl_tuning_file)
    if flags.gencode:
        generate_opencl_code(merged_opencl_tuning_file,
                             "LoadOpenCLParameter",
                             "LoadOpenCLParameter",
                             "mace/codegen/opencl/opencl_parameter.cc")
