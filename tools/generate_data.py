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
import sys
import numpy as np
import re
import common

# Validation Flow:
# 1. Generate input data
#    python generate_data.py \
#        --input_node input_node \
#        --input_shape 1,64,64,3 \
#        --input_file input_file \
#        --input_ranges -1,1


def generate_data(name, shape, input_file, tensor_range, input_data_type):
    np.random.seed()
    data = np.random.random(shape) * (tensor_range[1] - tensor_range[0]) \
        + tensor_range[0]
    input_file_name = common.formatted_file_name(input_file, name)
    print 'Generate input file: ', input_file_name
    if input_data_type == 'float32':
        np_data_type = np.float32
    elif input_data_type == 'int32':
        np_data_type = np.int32
    data.astype(np_data_type).tofile(input_file_name)


def generate_input_data(input_file, input_node, input_shape, input_ranges,
                        input_data_type):
    input_names = [name for name in input_node.split(',')]
    input_shapes = [shape for shape in input_shape.split(':')]

    if input_ranges:
        input_ranges = [r for r in input_ranges.split(':')]
    else:
        input_ranges = ["-1,1"] * len(input_names)
    if input_data_type:
        input_data_types = [data_type
                            for data_type in input_data_type.split(',')]
    else:
        input_data_types = ['float32'] * len(input_names)

    assert len(input_names) == len(input_shapes) == len(input_ranges) == len(input_data_types)  # noqa
    for i in range(len(input_names)):
        shape = [int(x) for x in input_shapes[i].split(',')]
        input_range = [float(x) for x in input_ranges[i].split(',')]
        generate_data(input_names[i], shape, input_file, input_range,
                      input_data_types[i])
    print "Generate input file done."


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--input_file", type=str, default="", help="input file.")
    parser.add_argument(
        "--input_node", type=str, default="input_node", help="input node")
    parser.add_argument(
        "--input_shape", type=str, default="1,64,64,3", help="input shape.")
    parser.add_argument(
        "--input_ranges", type=str, default="-1,1", help="input range.")
    parser.add_argument(
        "--input_data_type", type=str, default="", help="input range.")

    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    generate_input_data(FLAGS.input_file, FLAGS.input_node, FLAGS.input_shape,
                        FLAGS.input_ranges, FLAGS.input_data_type)
