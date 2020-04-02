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

import re
import os
import copy
import yaml
from enum import Enum

from utils.util import mace_check
from utils.util import MaceLogger
from py_proto import mace_pb2

CPP_KEYWORDS = [
    'alignas', 'alignof', 'and', 'and_eq', 'asm', 'atomic_cancel',
    'atomic_commit', 'atomic_noexcept', 'auto', 'bitand', 'bitor',
    'bool', 'break', 'case', 'catch', 'char', 'char16_t', 'char32_t',
    'class', 'compl', 'concept', 'const', 'constexpr', 'const_cast',
    'continue', 'co_await', 'co_return', 'co_yield', 'decltype', 'default',
    'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit',
    'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if',
    'import', 'inline', 'int', 'long', 'module', 'mutable', 'namespace',
    'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq',
    'private', 'protected', 'public', 'register', 'reinterpret_cast',
    'requires', 'return', 'short', 'signed', 'sizeof', 'static',
    'static_assert', 'static_cast', 'struct', 'switch', 'synchronized',
    'template', 'this', 'thread_local', 'throw', 'true', 'try', 'typedef',
    'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void',
    'volatile', 'wchar_t', 'while', 'xor', 'xor_eq', 'override', 'final',
    'transaction_safe', 'transaction_safe_dynamic', 'if', 'elif', 'else',
    'endif', 'defined', 'ifdef', 'ifndef', 'define', 'undef', 'include',
    'line', 'error', 'pragma',
]


def sanitize_load(s):
    # do not let yaml parse ON/OFF to boolean
    for w in ["ON", "OFF", "on", "off"]:
        s = re.sub(r":\s+" + w + "$", r": '" + w + "'", s)

    # sub ${} to env value
    s = re.sub(r"\${(\w+)}", lambda x: os.environ[x.group(1)], s)
    return yaml.load(s)


def parse(path):
    with open(path) as f:
        config = sanitize_load(f.read())

    return config


def parse_device_info(path):
    conf = parse(path)
    return conf["devices"]


class ModelKeys(object):
    platform = "platform"
    runtime = "runtime"
    models = 'models'
    graph_optimize_options = "graph_optimize_options"
    input_tensors = "input_tensors"
    input_shapes = "input_shapes"
    input_data_types = "input_data_types"
    input_data_formats = "input_data_formats"
    input_ranges = "input_ranges"
    output_tensors = "output_tensors"
    output_shapes = "output_shapes"
    output_data_types = "output_data_types"
    output_data_formats = "output_data_formats"
    check_tensors = "check_tensors"
    check_shapes = "check_shapes"
    model_file_path = "model_file_path"
    model_sha256_checksum = "model_sha256_checksum"
    weight_file_path = "weight_file_path"
    weight_sha256_checksum = "weight_sha256_checksum"
    quantize_range_file = "quantize_range_file"
    quantize = "quantize"
    quantize_large_weights = "quantize_large_weights"
    quantize_stat = "quantize_stat"
    change_concat_ranges = "change_concat_ranges"
    winograd = "winograd"
    cl_mem_type = "cl_mem_type"
    data_type = "data_type"
    subgraphs = "subgraphs"
    validation_inputs_data = "validation_inputs_data"


class DataFormat(Enum):
    NONE = 0
    NHWC = 1
    NCHW = 2
    HWIO = 100
    OIHW = 101
    HWOI = 102
    OHWI = 103
    AUTO = 1000


def parse_data_format(str):
    str = str.upper()
    mace_check(str in [e.name for e in DataFormat],
               "unknown data format %s" % str)
    return DataFormat[str]


class DeviceType(Enum):
    CPU = 0
    GPU = 2
    HEXAGON = 3
    HTA = 4
    APU = 5
    CPU_GPU = 100


DEVICE_MAP = {
    "cpu": DeviceType.CPU,
    "gpu": DeviceType.GPU,
    "hexagon": DeviceType.HEXAGON,
    "dsp": DeviceType.HEXAGON,
    "hta": DeviceType.HTA,
    "apu": DeviceType.APU,
    "cpu+gpu": DeviceType.CPU_GPU
}


def parse_device_type(str):
    mace_check(str in DEVICE_MAP, "unknown device %s" % str)
    return DEVICE_MAP[str]


class Platform(Enum):
    TENSORFLOW = 0
    CAFFE = 1
    ONNX = 2


def parse_platform(str):
    str = str.upper()
    mace_check(str in [e.name for e in Platform],
               "unknown platform %s" % str)
    return Platform[str]


DATA_TYPE_MAP = {
    'float32': mace_pb2.DT_FLOAT,
    'int32': mace_pb2.DT_INT32,
}


def parse_data_type(str):
    if str == "float32":
        return mace_pb2.DT_FLOAT
    elif str == "int32":
        return mace_pb2.DT_INT32
    else:
        mace_check(False, "data type %s not supported" % str)


def parse_internal_data_type(str):
    if str == 'fp32_fp32':
        return mace_pb2.DT_FLOAT
    elif str == 'bf16_fp32':
        return mace_pb2.DT_BFLOAT16
    else:
        return mace_pb2.DT_HALF


def to_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def parse_int_array(xs):
    if len(xs) is 0:
        return [1]
    return [int(x) for x in xs.split(",")]


def parse_float_array(xs):
    return [float(x) for x in xs.split(",")]


def normalize_model_config(conf):
    conf = copy.deepcopy(conf)
    if ModelKeys.subgraphs in conf:
        subgraph = conf[ModelKeys.subgraphs][0]
        del conf[ModelKeys.subgraphs]
        conf.update(subgraph)

    conf[ModelKeys.platform] = parse_platform(conf[ModelKeys.platform])
    conf[ModelKeys.runtime] = parse_device_type(conf[ModelKeys.runtime])

    if ModelKeys.quantize in conf and conf[ModelKeys.quantize] == 1:
        conf[ModelKeys.data_type] = mace_pb2.DT_FLOAT
    else:
        if ModelKeys.data_type in conf:
            conf[ModelKeys.data_type] = parse_internal_data_type(
                conf[ModelKeys.data_type])
        else:
            conf[ModelKeys.data_type] = mace_pb2.DT_HALF

    # parse input
    conf[ModelKeys.input_tensors] = to_list(conf[ModelKeys.input_tensors])
    conf[ModelKeys.input_tensors] = [str(i) for i in
                                     conf[ModelKeys.input_tensors]]
    input_count = len(conf[ModelKeys.input_tensors])
    conf[ModelKeys.input_shapes] = [parse_int_array(shape) for shape in
                                    to_list(conf[ModelKeys.input_shapes])]
    mace_check(
        len(conf[ModelKeys.input_shapes]) == input_count,
        "input node count and shape count do not match")

    input_data_types = [parse_data_type(dt) for dt in
                        to_list(conf.get(ModelKeys.input_data_types,
                                         ["float32"]))]

    if len(input_data_types) == 1 and input_count > 1:
        input_data_types = [input_data_types[0]] * input_count
    mace_check(len(input_data_types) == input_count,
               "the number of input_data_types should be "
               "the same as input tensors")
    conf[ModelKeys.input_data_types] = input_data_types

    input_data_formats = [parse_data_format(df) for df in
                          to_list(conf.get(ModelKeys.input_data_formats,
                                           ["NHWC"]))]
    if len(input_data_formats) == 1 and input_count > 1:
        input_data_formats = [input_data_formats[0]] * input_count
    mace_check(len(input_data_formats) == input_count,
               "the number of input_data_formats should be "
               "the same as input tensors")
    conf[ModelKeys.input_data_formats] = input_data_formats

    input_ranges = [parse_float_array(r) for r in
                    to_list(conf.get(ModelKeys.input_ranges,
                                     ["-1.0,1.0"]))]
    if len(input_ranges) == 1 and input_count > 1:
        input_ranges = [input_ranges[0]] * input_count
    mace_check(len(input_ranges) == input_count,
               "the number of input_ranges should be "
               "the same as input tensors")
    conf[ModelKeys.input_ranges] = input_ranges

    # parse output
    conf[ModelKeys.output_tensors] = to_list(conf[ModelKeys.output_tensors])
    conf[ModelKeys.output_tensors] = [str(i) for i in
                                      conf[ModelKeys.output_tensors]]
    output_count = len(conf[ModelKeys.output_tensors])
    conf[ModelKeys.output_shapes] = [parse_int_array(shape) for shape in
                                     to_list(conf[ModelKeys.output_shapes])]
    mace_check(len(conf[ModelKeys.output_tensors]) == output_count,
               "output node count and shape count do not match")

    output_data_types = [parse_data_type(dt) for dt in
                         to_list(conf.get(ModelKeys.output_data_types,
                                          ["float32"]))]
    if len(output_data_types) == 1 and output_count > 1:
        output_data_types = [output_data_types[0]] * output_count
    mace_check(len(output_data_types) == output_count,
               "the number of output_data_types should be "
               "the same as output tensors")
    conf[ModelKeys.output_data_types] = output_data_types

    output_data_formats = [parse_data_format(df) for df in
                           to_list(conf.get(ModelKeys.output_data_formats,
                                            ["NHWC"]))]
    if len(output_data_formats) == 1 and output_count > 1:
        output_data_formats = [output_data_formats[0]] * output_count
    mace_check(len(output_data_formats) == output_count,
               "the number of output_data_formats should be "
               "the same as output tensors")
    conf[ModelKeys.output_data_formats] = output_data_formats

    if ModelKeys.check_tensors in conf:
        conf[ModelKeys.check_tensors] = to_list(conf[ModelKeys.check_tensors])
        conf[ModelKeys.check_shapes] = [parse_int_array(shape) for shape in
                                        to_list(conf[ModelKeys.check_shapes])]
        mace_check(len(conf[ModelKeys.check_tensors]) == len(
            conf[ModelKeys.check_shapes]),
                   "check tensors count and shape count do not match.")

    MaceLogger.summary(conf)

    return conf
