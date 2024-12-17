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

from utils import util
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
    ret = None
    try: 
        ret = yaml.load(s)
    except:
        ret = yaml.full_load(s)
    return ret


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
    output_aliases = "output_aliases"
    input_aliases = "input_aliases"
    model_file_path = "model_file_path"
    model_sha256_checksum = "model_sha256_checksum"
    weight_file_path = "weight_file_path"
    weight_sha256_checksum = "weight_sha256_checksum"
    quantize_range_file = "quantize_range_file"
    quantize = "quantize"
    quantize_schema = "quantize_schema"
    quantize_large_weights = "quantize_large_weights"
    quantize_stat = "quantize_stat"
    change_concat_ranges = "change_concat_ranges"
    winograd = "winograd"
    cl_mem_type = "cl_mem_type"
    data_type = "data_type"
    subgraphs = "subgraphs"
    default_graph = 'default_graph'
    order = 'order'
    validation_inputs_data = "validation_inputs_data"
    validation_outputs_data = 'validation_outputs_data'


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


# must be same as MemoryType in mace.h
class MemoryType(Enum):
    CPU_BUFFER = 0
    GPU_BUFFER = 1
    GPU_IMAGE = 2
    MEMORY_NONE = 10000


class DeviceType(Enum):
    CPU = 0
    GPU = 2
    HEXAGON = 3
    HTA = 4
    APU = 5
    HTP = 6
    CPU_GPU = 100


DEVICE_MAP = {
    "cpu": DeviceType.CPU,
    "gpu": DeviceType.GPU,
    "hexagon": DeviceType.HEXAGON,
    "dsp": DeviceType.HEXAGON,
    "hta": DeviceType.HTA,
    "apu": DeviceType.APU,
    "cpu+gpu": DeviceType.GPU,
    "htp": DeviceType.HTP,
}


def parse_device_type(str):
    mace_check(str in DEVICE_MAP, "unknown device %s" % str)
    return DEVICE_MAP[str]


class Platform(Enum):
    TENSORFLOW = 0
    CAFFE = 1
    ONNX = 2
    MEGENGINE = 3
    KERAS = 4
    PYTORCH = 5


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
    if str == "float16":
        return mace_pb2.DT_FLOAT16
    elif str == "int32":
        return mace_pb2.DT_INT32
    elif str == "int16":
        return mace_pb2.DT_INT16
    elif str == "uint8":
        return mace_pb2.DT_UINT8
    else:
        mace_check(False, "data type %s not supported" % str)


def parse_internal_data_type(str):
    if str == 'fp32_fp32':
        return mace_pb2.DT_FLOAT
    elif str == 'bf16_fp32':
        return mace_pb2.DT_BFLOAT16
    elif str == 'fp16_fp16':
        return mace_pb2.DT_FLOAT16
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


def normalize_input_data_types(conf, input_count):
    default_input_dt = conf.get(ModelKeys.data_type, mace_pb2.DT_FLOAT)
    if default_input_dt == mace_pb2.DT_HALF:
        default_input_dt = mace_pb2.DT_FLOAT  # Compatible with old version
    conf_input_dts = to_list(conf.get(ModelKeys.input_data_types, []))
    if len(conf_input_dts) == 0:
        input_data_types = [default_input_dt]
    else:
        input_data_types = [parse_data_type(dt) for dt in conf_input_dts]

    if len(input_data_types) == 1 and input_count > 1:
        input_data_types = [input_data_types[0]] * input_count
    mace_check(len(input_data_types) == input_count,
               "the number of input_data_types should be "
               "the same as input tensors")
    conf[ModelKeys.input_data_types] = input_data_types


def normalize_output_data_types(conf, output_count):
    default_output_dt = conf.get(ModelKeys.data_type, mace_pb2.DT_FLOAT)
    if default_output_dt == mace_pb2.DT_HALF:
        default_output_dt = mace_pb2.DT_FLOAT  # Compatible with old version
    conf_output_dts = to_list(conf.get(ModelKeys.output_data_types, []))
    if len(conf_output_dts) == 0:
        output_data_types = [default_output_dt]
    else:
        output_data_types = [parse_data_type(dt) for dt in conf_output_dts]

    if len(output_data_types) == 1 and output_count > 1:
        output_data_types = [output_data_types[0]] * output_count
    mace_check(len(output_data_types) == output_count,
               "the number of output_data_types should be "
               "the same as output tensors")
    conf[ModelKeys.output_data_types] = output_data_types


def normalize_graph_config(conf, model_output, org_model_dir):
    conf = copy.deepcopy(conf)
    if ModelKeys.platform in conf:
        conf[ModelKeys.platform] = parse_platform(conf[ModelKeys.platform])
    if ModelKeys.model_file_path in conf and org_model_dir is not None:
        model_file = util.download_or_get_model(
            conf[ModelKeys.model_file_path],
            conf[ModelKeys.model_sha256_checksum], org_model_dir)
        conf[ModelKeys.model_file_path] = model_file
    if ModelKeys.weight_file_path in conf:
        weight_file = util.download_or_get_model(
            conf[ModelKeys.weight_file_path],
            conf[ModelKeys.weight_sha256_checksum], "/tmp/")
        conf[ModelKeys.weight_file_path] = weight_file

    if ModelKeys.runtime in conf:
        conf[ModelKeys.runtime] = parse_device_type(conf[ModelKeys.runtime])

    if ModelKeys.data_type in conf:
        conf[ModelKeys.data_type] = parse_internal_data_type(
            conf[ModelKeys.data_type])

    # TODO: remove the following after quantize tool is made
    if ModelKeys.quantize_range_file in conf and model_output is not None:
        range_file = util.download_or_get_model(
            conf[ModelKeys.quantize_range_file],
            "", model_output)
        conf[ModelKeys.quantize_range_file] = range_file

    input_count = 0
    input_data_formats = []
    input_ranges = []
    if ModelKeys.input_tensors in conf:
        conf[ModelKeys.input_tensors] = to_list(conf[ModelKeys.input_tensors])
        conf[ModelKeys.input_tensors] = [str(i) for i in
                                         conf[ModelKeys.input_tensors]]
        input_count = len(conf[ModelKeys.input_tensors])
        input_data_formats = [parse_data_format(df) for df in
                              to_list(conf.get(ModelKeys.input_data_formats,
                                               ["NHWC"]))]
        input_ranges = [parse_float_array(r) for r in
                        to_list(conf.get(ModelKeys.input_ranges,
                                         ["-1.0,1.0"]))]
        normalize_input_data_types(conf, input_count)

    if ModelKeys.input_shapes in conf:
        conf[ModelKeys.input_shapes] = [parse_int_array(shape) for shape in
                                        to_list(conf[ModelKeys.input_shapes])]
        mace_check(
            len(conf[ModelKeys.input_shapes]) == input_count,
            "input node count and shape count do not match")

    if len(input_data_formats) == 1 and input_count > 1:
        input_data_formats = [input_data_formats[0]] * input_count
    mace_check(len(input_data_formats) == input_count,
               "the number of input_data_formats should be "
               "the same as input tensors")
    conf[ModelKeys.input_data_formats] = input_data_formats

    if len(input_ranges) == 1 and input_count > 1:
        input_ranges = [input_ranges[0]] * input_count
    mace_check(len(input_ranges) == input_count,
               "the number of input_ranges should be "
               "the same as input tensors")
    conf[ModelKeys.input_ranges] = input_ranges

    # parse output
    output_count = 0
    output_data_types = []
    output_data_formats = []
    if ModelKeys.output_tensors in conf:
        conf[ModelKeys.output_tensors] = \
            to_list(conf[ModelKeys.output_tensors])
        conf[ModelKeys.output_tensors] = [str(i) for i in
                                          conf[ModelKeys.output_tensors]]
        output_count = len(conf[ModelKeys.output_tensors])
        output_data_formats = [parse_data_format(df) for df in
                               to_list(conf.get(ModelKeys.output_data_formats,
                                                ["NHWC"]))]
        normalize_output_data_types(conf, output_count)

    if ModelKeys.output_shapes in conf:
        conf[ModelKeys.output_shapes] = [
            parse_int_array(shape) for shape in
            to_list(conf[ModelKeys.output_shapes])]
        mace_check(len(conf[ModelKeys.output_tensors]) == output_count,
                   "output node count and shape count do not match")

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
    return conf


def set_default_config_value(nor_subgraph, model):
    if ModelKeys.quantize in model and model[ModelKeys.quantize] == 1:
        model[ModelKeys.data_type] = mace_pb2.DT_FLOAT
    if ModelKeys.quantize in nor_subgraph and \
            nor_subgraph[ModelKeys.quantize] == 1:
        nor_subgraph[ModelKeys.data_type] = mace_pb2.DT_FLOAT
    elif ModelKeys.data_type not in nor_subgraph:
        if ModelKeys.data_type in model:
            nor_subgraph[ModelKeys.data_type] = model[ModelKeys.data_type]
        elif ModelKeys.quantize in model and model[ModelKeys.quantize] == 1:
            nor_subgraph[ModelKeys.data_type] = mace_pb2.DT_FLOAT
        else:
            nor_subgraph[ModelKeys.data_type] = mace_pb2.DT_HALF


def normalize_model_config(conf, model_output=None, org_model_dir=None):
    conf = normalize_graph_config(conf, model_output, org_model_dir)
    if ModelKeys.subgraphs in conf:
        nor_subgraphs = {}
        if isinstance(conf[ModelKeys.subgraphs], list):
            nor_subgraph = normalize_graph_config(conf[ModelKeys.subgraphs][0],
                                                  model_output, org_model_dir)
            conf[ModelKeys.input_tensors] = \
                nor_subgraph[ModelKeys.input_tensors]
            conf[ModelKeys.output_tensors] = \
                nor_subgraph[ModelKeys.output_tensors]
            if ModelKeys.validation_inputs_data in nor_subgraph:
                conf[ModelKeys.validation_inputs_data] = \
                    nor_subgraph[ModelKeys.validation_inputs_data]
            if ModelKeys.validation_outputs_data in nor_subgraph:
                conf[ModelKeys.validation_outputs_data] = \
                    nor_subgraph[ModelKeys.validation_outputs_data]
            set_default_config_value(nor_subgraph, conf)
            nor_subgraphs[ModelKeys.default_graph] = nor_subgraph
        else:
            for graph_name, subgraph in conf[ModelKeys.subgraphs].items():
                nor_subgraph = normalize_graph_config(subgraph, model_output,
                                                      org_model_dir)
                set_default_config_value(nor_subgraph, conf)
                nor_subgraphs[graph_name] = nor_subgraph

        conf[ModelKeys.subgraphs] = nor_subgraphs

        model_base_conf = copy.deepcopy(conf)
        del model_base_conf[ModelKeys.subgraphs]
        subgraphs = conf[ModelKeys.subgraphs]
        for net_name, subgraph in subgraphs.items():
            net_conf = copy.deepcopy(model_base_conf)
            net_conf.update(subgraph)
            subgraphs[net_name] = net_conf

    MaceLogger.summary(conf)
    return conf


def find_input_tensors_info(subgraphs, tensor_names):
    tensors_info = {}
    all_tensor_names = []
    all_tensor_shapes = []
    all_data_formats = []
    all_data_types = []
    all_ranges = []
    for (subname, subgraph) in subgraphs.items():
        all_tensor_names.extend(subgraph[ModelKeys.input_tensors])
        all_tensor_shapes.extend(subgraph[ModelKeys.input_shapes])
        all_data_formats.extend(subgraph[ModelKeys.input_data_formats])
        all_data_types.extend(subgraph[ModelKeys.input_data_types])
        if ModelKeys.input_ranges in subgraph:
            all_ranges.extend(subgraph[ModelKeys.input_ranges])
        else:
            all_ranges.extend([""] * len(subgraph[ModelKeys.input_tensors]))
    name_id = {}
    for i in range(len(all_tensor_names)):
        name_id[all_tensor_names[i]] = i
    tensors_info[ModelKeys.input_tensors] = []
    tensors_info[ModelKeys.input_shapes] = []
    tensors_info[ModelKeys.input_data_formats] = []
    tensors_info[ModelKeys.input_data_types] = []
    tensors_info[ModelKeys.input_ranges] = []
    for tensor_name in tensor_names:
        i = name_id[tensor_name]
        tensors_info[ModelKeys.input_tensors].append(tensor_name)
        tensors_info[ModelKeys.input_shapes].append(all_tensor_shapes[i])
        tensors_info[ModelKeys.input_data_formats].append(all_data_formats[i])
        tensors_info[ModelKeys.input_data_types].append(all_data_types[i])
        tensors_info[ModelKeys.input_ranges].append(all_ranges[i])
    return tensors_info


def find_output_tensors_info(subgraphs, tensor_names):
    tensors_info = {}
    all_tensor_names = []
    all_tensor_shapes = []
    all_data_formats = []
    all_data_types = []
    all_check_tensor_names = []
    all_check_tensor_shapes = []
    for (subname, subgraph) in subgraphs.items():
        all_tensor_names.extend(subgraph[ModelKeys.output_tensors])
        all_tensor_shapes.extend(subgraph[ModelKeys.output_shapes])
        all_data_formats.extend(subgraph[ModelKeys.output_data_formats])
        all_data_types.extend(subgraph[ModelKeys.output_data_types])
        output_num = len(subgraph[ModelKeys.output_tensors])
        if ModelKeys.check_tensors in subgraph:
            all_check_tensor_names.extend(subgraph[ModelKeys.check_tensors])
        else:
            all_check_tensor_names.extend([None] * output_num)
        if ModelKeys.check_shapes in subgraph:
            all_check_tensor_shapes.extend(subgraph[ModelKeys.check_shapes])
        else:
            all_check_tensor_shapes.extend([None] * output_num)

    name_id = {}
    for i in range(len(all_tensor_names)):
        name_id[all_tensor_names[i]] = i
    tensors_info[ModelKeys.output_tensors] = []
    tensors_info[ModelKeys.output_shapes] = []
    tensors_info[ModelKeys.output_data_formats] = []
    tensors_info[ModelKeys.output_data_types] = []
    tensors_info[ModelKeys.check_tensors] = []
    tensors_info[ModelKeys.check_shapes] = []
    for tensor_name in tensor_names:
        tensor_name = str(tensor_name)
        i = name_id[tensor_name]
        tensors_info[ModelKeys.output_tensors].append(tensor_name)
        tensors_info[ModelKeys.output_shapes].append(all_tensor_shapes[i])
        tensors_info[ModelKeys.output_data_formats].append(all_data_formats[i])
        tensors_info[ModelKeys.output_data_types].append(all_data_types[i])
        tensors_info[ModelKeys.check_tensors].append(all_check_tensor_names[i])
        tensors_info[ModelKeys.check_shapes].append(all_check_tensor_shapes[i])
    return tensors_info
