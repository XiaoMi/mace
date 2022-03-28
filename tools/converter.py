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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sh
import sys
import time
import yaml
import sh_commands
from enum import Enum

sys.path.insert(0, "tools/python")  # noqa
from common import *
from device import DeviceWrapper, DeviceManager
from utils import config_parser
import convert
import encrypt

from dana.dana_util import DanaUtil

################################
# set environment
################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################
# common definitions
################################

ABITypeStrs = [
    'armeabi-v7a',
    'arm64-v8a',
    'arm64',
    'armhf',
    'host',
]

ModelFormatStrs = [
    "file",
    "code",
]

PlatformTypeStrs = [
    "tensorflow",
    "caffe",
    "onnx",
    "megengine",
    "keras",
    "pytorch",
]
PlatformType = Enum('PlatformType', [(ele, ele) for ele in PlatformTypeStrs],
                    type=str)

RuntimeTypeStrs = [
    "cpu",
    "gpu",
    "dsp",
    "hta",
    "apu",
    "htp"
]

InOutDataTypeStrs = [
    "int32",
    "float32",
    "float16",
    "bfloat16",
    "int16",
    "uint8",
]

InOutDataType = Enum('InputDataType',
                     [(ele, ele) for ele in InOutDataTypeStrs],
                     type=str)

FPDataTypeStrs = [
    "fp16_fp32",
    "fp32_fp32",
    "bf16_fp32",
    "fp16_fp16",
]

FPDataType = Enum('GPUDataType', [(ele, ele) for ele in FPDataTypeStrs],
                  type=str)

DSPDataTypeStrs = [
    "uint8",
]

DSPDataType = Enum('DSPDataType', [(ele, ele) for ele in DSPDataTypeStrs],
                   type=str)

APUDataTypeStrs = [
    "fp16_fp32",
    "fp32_fp32",
    "uint8",
]

APUDataType = Enum('APUDataType', [(ele, ele) for ele in APUDataTypeStrs],
                   type=str)

WinogradParameters = [0, 2, 4]

DataFormatStrs = [
    "NONE",
    "NHWC",
    "NCHW",
    "OIHW",
]


class DefaultValues(object):
    mace_lib_type = MACELibType.static
    num_threads = -1,
    cpu_affinity_policy = 1,
    gpu_perf_hint = 3,
    gpu_priority_hint = 3,
    accelerator_cache_policy = 0,
    apu_boost_hint = 100,
    apu_preference_hint = 1,


class ValidationThreshold(object):
    cpu_threshold = 0.999,
    gpu_threshold = 0.995,
    quantize_threshold = 0.980,


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


################################
# common functions
################################
def parse_device_type(runtime):
    device_type = ""

    if runtime == RuntimeType.dsp:
        device_type = DeviceType.HEXAGON
    elif runtime == RuntimeType.hta:
        device_type = DeviceType.HTA
    elif runtime == RuntimeType.htp:
        device_type = DeviceType.HTP
    elif runtime == RuntimeType.gpu:
        device_type = DeviceType.GPU
    elif runtime == RuntimeType.cpu:
        device_type = DeviceType.CPU
    elif runtime == RuntimeType.apu:
        device_type = DeviceType.APU
    elif runtime == RuntimeType.cpu_gpu:
        device_type = DeviceType.GPU

    return device_type


def get_data_type_and_io_types(configs):
    data_types = []
    input_types = []
    output_types = []
    for (model_name, model_config) in configs[YAMLKeyword.models].items():
        default_data_type = model_config[YAMLKeyword.data_type] \
            if YAMLKeyword.data_type in model_config else FPDataType.fp16_fp32
        default_input_types = InOutDataType.float32
        if YAMLKeyword.input_data_types in model_config:
            default_input_types = model_config[YAMLKeyword.input_data_types]
        default_output_types = InOutDataType.float32
        if YAMLKeyword.output_data_types in model_config:
            default_output_types = model_config[YAMLKeyword.output_data_types]

        subgraphs = model_config[YAMLKeyword.subgraphs]
        for graph_name, graph_config in subgraphs.items():
            dtype = graph_config.get(YAMLKeyword.data_type, default_data_type)
            data_types.append(dtype)
            input_type_array = graph_config.get(YAMLKeyword.input_data_types,
                                                default_input_types)
            input_types.extend(input_type_array)
            output_type_array = graph_config.get(YAMLKeyword.output_data_types,
                                                 default_output_types)
            output_types.extend(output_type_array)

    return (data_types, input_types, output_types)


def data_type_enabled(configs, data_type, io_type):
    (data_types, input_types, output_types) = \
        get_data_type_and_io_types(configs)
    if (data_type in data_types or io_type in input_types or
            io_type in output_types):
        return True
    return False


def bfloat16_enabled(configs):
    return data_type_enabled(configs, FPDataType.bf16_fp32,
                             InOutDataType.bfloat16)


def fp16_enabled(configs):
    return data_type_enabled(configs, FPDataType.fp16_fp16,
                             InOutDataType.float16)


def get_runtimes(configs):
    runtime_list = []
    for (model_name, model_config) in configs[YAMLKeyword.models].items():
        subgraphs = model_config[YAMLKeyword.subgraphs]
        default_rt = model_config[YAMLKeyword.runtime] \
            if YAMLKeyword.runtime in model_config else RuntimeType.cpu
        for graph_name, graph_config in subgraphs.items():
            model_runtime = graph_config.get(YAMLKeyword.runtime, default_rt)
            runtime_list.append(model_runtime)
    return runtime_list


def hexagon_enabled(configs):
    if RuntimeType.dsp in get_runtimes(configs):
        return True
    return False


def hta_enabled(configs):
    if RuntimeType.hta in get_runtimes(configs):
        return True
    return False


def htp_enabled(configs):
    return RuntimeType.htp in get_runtimes(configs)


def qnn_enabled(configs):
    accelerator_apis = []
    for model_name in configs[YAMLKeyword.models]:
        accelerator_api = \
            configs[YAMLKeyword.models][model_name].get(
                YAMLKeyword.accelerator_api, "")
        accelerator_apis.append(accelerator_api.lower())

    return 'qnn' in accelerator_apis or htp_enabled(configs)


def apu_enabled(configs):
    if RuntimeType.apu in get_runtimes(configs):
        return True
    return False


def opencl_enabled(configs):
    runtime_list = get_runtimes(configs)
    if RuntimeType.gpu in runtime_list or RuntimeType.cpu_gpu in runtime_list \
            or RuntimeType.hta in runtime_list:
        return True
    return False


def quantize_enabled(configs):
    for (model_name, model_config) in configs[YAMLKeyword.models].items():
        quantize = model_config.get(YAMLKeyword.quantize, 0)
        if quantize == 1:
            return True
        subgraphs = model_config[YAMLKeyword.subgraphs]
        for graph_name, graph_config in subgraphs.items():
            quantize = graph_config.get(YAMLKeyword.quantize, 0)
            if quantize == 1:
                return True
    return False


def get_symbol_hidden_mode(debug_mode, mace_lib_type=None):
    if not mace_lib_type:
        return True
    if debug_mode or mace_lib_type == MACELibType.dynamic:
        return False
    else:
        return True


def md5sum(str):
    md5 = hashlib.md5()
    md5.update(str.encode('utf-8'))
    return md5.hexdigest()


def sha256_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_file(url, dst, num_retries=3):
    from six.moves import urllib

    try:
        urllib.request.urlretrieve(url, dst)
        MaceLogger.info('\nDownloaded successfully.')
    except (urllib.error.ContentTooShortError, urllib.error.HTTPError,
            urllib.error.URLError) as e:
        MaceLogger.warning('Download error:' + str(e))
        if num_retries > 0:
            return download_file(url, dst, num_retries - 1)
        else:
            return False
    return True


def get_model_files(model_config, model_output_dir):
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    model_file_path = model_config[YAMLKeyword.model_file_path]
    model_sha256_checksum = model_config[YAMLKeyword.model_sha256_checksum]
    weight_file_path = model_config.get(YAMLKeyword.weight_file_path, "")
    weight_sha256_checksum = \
        model_config.get(YAMLKeyword.weight_sha256_checksum, "")
    quantize_range_file_path = \
        model_config.get(YAMLKeyword.quantize_range_file, "")
    model_file = model_file_path
    weight_file = weight_file_path
    quantize_range_file = quantize_range_file_path

    if model_file_path.startswith("http://") or \
            model_file_path.startswith("https://"):
        model_file = model_output_dir + "/" + md5sum(model_file_path) + ".pb"
        if not os.path.exists(model_file) or \
                sha256_checksum(model_file) != model_sha256_checksum:
            MaceLogger.info("Downloading model, please wait ...")
            if not download_file(model_file_path, model_file):
                MaceLogger.error(ModuleName.MODEL_CONVERTER,
                                 "Model download failed.")
        model_config[YAMLKeyword.model_file_path] = model_file

    if sha256_checksum(model_file) != model_sha256_checksum:
        error_info = model_file_path + \
                     " model file sha256checksum not match " + \
                     model_sha256_checksum
        MaceLogger.error(ModuleName.MODEL_CONVERTER, error_info)

    if weight_file_path.startswith("http://") or \
            weight_file_path.startswith("https://"):
        weight_file = \
            model_output_dir + "/" + md5sum(weight_file_path) + ".caffemodel"
        if not os.path.exists(weight_file) or \
                sha256_checksum(weight_file) != weight_sha256_checksum:
            MaceLogger.info("Downloading model weight, please wait ...")
            if not download_file(weight_file_path, weight_file):
                MaceLogger.error(ModuleName.MODEL_CONVERTER,
                                 "Model download failed.")
    model_config[YAMLKeyword.weight_file_path] = weight_file

    if weight_file:
        if sha256_checksum(weight_file) != weight_sha256_checksum:
            error_info = weight_file_path + \
                         " weight file sha256checksum not match " + \
                         weight_sha256_checksum
            MaceLogger.error(ModuleName.MODEL_CONVERTER, error_info)

    if quantize_range_file_path.startswith("http://") or \
            quantize_range_file_path.startswith("https://"):
        quantize_range_file = \
            model_output_dir + "/" + md5sum(quantize_range_file_path) \
            + ".range"
        if not download_file(quantize_range_file_path, quantize_range_file):
            MaceLogger.error(ModuleName.MODEL_CONVERTER,
                             "Model range file download failed.")
    model_config[YAMLKeyword.quantize_range_file] = quantize_range_file


def get_data_type(parent_config, runtime):
    data_type = parent_config.get(YAMLKeyword.data_type, "")
    if runtime == RuntimeType.dsp:
        if len(data_type) > 0:
            mace_check(data_type in DSPDataTypeStrs,
                       ModuleName.YAML_CONFIG,
                       "'data_type' must be in " + str(DSPDataTypeStrs)
                       + " for dsp runtime")
        else:
            parent_config[YAMLKeyword.data_type] = \
                DSPDataType.uint8.value
    elif runtime == RuntimeType.apu:
        if len(data_type) > 0:
            mace_check(data_type in APUDataTypeStrs,
                       ModuleName.YAML_CONFIG,
                       "'data_type %s' must be in %s for apu runtime" %
                       (data_type, str(APUDataTypeStrs)))
        else:
            parent_config[YAMLKeyword.data_type] = \
                APUDataType.uint8.value
    else:
        if len(data_type) > 0:
            mace_check(data_type in FPDataTypeStrs,
                       ModuleName.YAML_CONFIG,
                       "'data_type' must be in " + str(FPDataTypeStrs)
                       + " for cpu runtime")
        else:
            if runtime == RuntimeType.cpu:
                parent_config[YAMLKeyword.data_type] = \
                    FPDataType.fp32_fp32.value
            else:
                parent_config[YAMLKeyword.data_type] = \
                    FPDataType.fp16_fp32.value
    return data_type


def format_model_config(flags):
    with open(flags.config) as f:
        configs = yaml.load(f)

    library_name = configs.get(YAMLKeyword.library_name, "")
    mace_check(len(library_name) > 0,
               ModuleName.YAML_CONFIG, "library name should not be empty")

    if flags.target_abis:
        target_abis = flags.target_abis.split(',')
    else:
        target_abis = configs.get(YAMLKeyword.target_abis, [])
    mace_check((isinstance(target_abis, list) and len(target_abis) > 0),
               ModuleName.YAML_CONFIG, "target_abis list is needed")
    configs[YAMLKeyword.target_abis] = target_abis
    for abi in target_abis:
        mace_check(abi in ABITypeStrs,
                   ModuleName.YAML_CONFIG,
                   "target_abis must be in " + str(ABITypeStrs))

    target_socs = configs.get(YAMLKeyword.target_socs, "")
    if flags.target_socs and flags.target_socs != TargetSOCTag.random \
            and flags.target_socs != TargetSOCTag.all:
        configs[YAMLKeyword.target_socs] = \
            [soc.lower() for soc in flags.target_socs.split(',')]
    elif not target_socs:
        configs[YAMLKeyword.target_socs] = []
    elif not isinstance(target_socs, list):
        configs[YAMLKeyword.target_socs] = [target_socs]

    configs[YAMLKeyword.target_socs] = \
        [soc.lower() for soc in configs[YAMLKeyword.target_socs]]

    if ABIType.armeabi_v7a in target_abis \
            or ABIType.arm64_v8a in target_abis:
        available_socs = sh_commands.adb_get_all_socs()
        target_socs = configs[YAMLKeyword.target_socs]
        if TargetSOCTag.all in target_socs:
            mace_check(available_socs,
                       ModuleName.YAML_CONFIG,
                       "Android abi is listed in config file and "
                       "build for all SOCs plugged in computer, "
                       "But no android phone found, "
                       "you at least plug in one phone")
        else:
            for soc in target_socs:
                mace_check(soc in available_socs, ModuleName.YAML_CONFIG,
                           "Build specified SOC library, you must plug in a "
                           "phone using the SOC(%s), the current socs: %s" %
                           (soc, available_socs))

    if flags.model_graph_format:
        model_graph_format = flags.model_graph_format
    else:
        model_graph_format = configs.get(YAMLKeyword.model_graph_format, "")
    mace_check(model_graph_format in ModelFormatStrs,
               ModuleName.YAML_CONFIG,
               'You must set model_graph_format and '
               "model_graph_format must be in " + str(ModelFormatStrs))
    configs[YAMLKeyword.model_graph_format] = model_graph_format
    if flags.model_data_format:
        model_data_format = flags.model_data_format
    else:
        model_data_format = configs.get(YAMLKeyword.model_data_format, "")
    configs[YAMLKeyword.model_data_format] = model_data_format
    mace_check(model_data_format in ModelFormatStrs,
               ModuleName.YAML_CONFIG,
               'You must set model_data_format and '
               "model_data_format must be in " + str(ModelFormatStrs))

    mace_check(not (model_graph_format == ModelFormat.file
                    and model_data_format == ModelFormat.code),
               ModuleName.YAML_CONFIG,
               "If model_graph format is 'file',"
               " the model_data_format must be 'file' too")

    model_names = configs.get(YAMLKeyword.models, [])
    mace_check(len(model_names) > 0, ModuleName.YAML_CONFIG,
               "no model found in config file")

    model_name_reg = re.compile(r'^[a-zA-Z0-9_]+$')
    for model_name in model_names:
        # check model_name legality
        mace_check(model_name not in CPP_KEYWORDS,
                   ModuleName.YAML_CONFIG,
                   "model name should not be c++ keyword.")
        mace_check((model_name[0] == '_' or model_name[0].isalpha())
                   and bool(model_name_reg.match(model_name)),
                   ModuleName.YAML_CONFIG,
                   "model name should Meet the c++ naming convention"
                   " which start with '_' or alpha"
                   " and only contain alpha, number and '_'")

        model_config = configs[YAMLKeyword.models][model_name]
        platform = model_config.get(YAMLKeyword.platform, "")
        mace_check(platform in PlatformTypeStrs,
                   ModuleName.YAML_CONFIG,
                   "'platform' must be in " + str(PlatformTypeStrs))

        for key in [YAMLKeyword.model_file_path,
                    YAMLKeyword.model_sha256_checksum]:
            value = model_config.get(key, "")
            mace_check(value != "", ModuleName.YAML_CONFIG,
                       "'%s' is necessary" % key)

        weight_file_path = model_config.get(YAMLKeyword.weight_file_path, "")
        if weight_file_path:
            weight_checksum = \
                model_config.get(YAMLKeyword.weight_sha256_checksum, "")
            mace_check(weight_checksum != "", ModuleName.YAML_CONFIG,
                       "'%s' is necessary" %
                       YAMLKeyword.weight_sha256_checksum)
        else:
            model_config[YAMLKeyword.weight_sha256_checksum] = ""

        get_model_files(model_config, BUILD_DOWNLOADS_DIR)

        subgraphs = model_config.get(YAMLKeyword.subgraphs, "")
        mace_check(len(subgraphs) > 0, ModuleName.YAML_CONFIG,
                   "at least one subgraph is needed")

        if isinstance(subgraphs, list):
            graphs_dict = {}
            model_config[YAMLKeyword.input_tensors] = \
                config_parser.to_list(subgraphs[0][YAMLKeyword.input_tensors])
            model_config[YAMLKeyword.output_tensors] = \
                config_parser.to_list(subgraphs[0][YAMLKeyword.output_tensors])
            model_config[YAMLKeyword.validation_inputs_data] = \
                config_parser.to_list(
                    subgraphs[0].get(YAMLKeyword.validation_inputs_data, []))
            model_config[YAMLKeyword.validation_outputs_data] = \
                config_parser.to_list(
                    subgraphs[0].get(YAMLKeyword.validation_outputs_data, []))
            graphs_dict[YAMLKeyword.default_graph] = subgraphs[0]
            subgraphs = graphs_dict
            model_config[YAMLKeyword.subgraphs] = subgraphs

        for graph_name, graph_config in subgraphs.items():
            runtime = DeviceWrapper.get_graph_runtime(
                graph_config, model_config, target_abis)
            get_data_type(model_config, runtime)

        input_ranges = model_config.get(
            YAMLKeyword.input_ranges, [])
        if not isinstance(input_ranges, list):
            model_config[YAMLKeyword.input_ranges] = [input_ranges]
        else:
            model_config[YAMLKeyword.input_ranges] = input_ranges
        model_config[YAMLKeyword.input_ranges] = \
            [str(v) for v in model_config[YAMLKeyword.input_ranges]]

        accuracy_validation_script = model_config.get(
            YAMLKeyword.accuracy_validation_script, "")
        if isinstance(accuracy_validation_script, list):
            mace_check(len(accuracy_validation_script) == 1,
                       ModuleName.YAML_CONFIG,
                       "Only support one accuracy validation script")
            accuracy_validation_script = accuracy_validation_script[0]
        model_config[YAMLKeyword.accuracy_validation_script] = \
            accuracy_validation_script
        validation_inputs_data = model_config.get(
            YAMLKeyword.validation_inputs_data, [])
        if not isinstance(validation_inputs_data, list):
            model_config[YAMLKeyword.validation_inputs_data] = [
                validation_inputs_data]
        else:
            model_config[YAMLKeyword.validation_inputs_data] = \
                validation_inputs_data

        for key in [YAMLKeyword.check_tensors, YAMLKeyword.check_shapes]:
            value = model_config.get(key, None)
            if value:
                if not isinstance(value, list):
                    model_config[key] = [value]
                model_config[key] = [str(v) for v in subgraph[key]]
            else:
                model_config[key] = []

        validation_threshold = model_config.get(
            YAMLKeyword.validation_threshold, {})
        if not isinstance(validation_threshold, dict):
            raise argparse.ArgumentTypeError(
                'similarity threshold must be a dict.')
        threshold_dict = {
            DeviceType.CPU: ValidationThreshold.cpu_threshold,
            DeviceType.GPU: ValidationThreshold.gpu_threshold,
            DeviceType.HEXAGON: ValidationThreshold.quantize_threshold,
            DeviceType.HTA: ValidationThreshold.quantize_threshold,
            DeviceType.APU: ValidationThreshold.quantize_threshold,
            DeviceType.HTP: ValidationThreshold.quantize_threshold,
            DeviceType.QUANTIZE: ValidationThreshold.quantize_threshold,
        }
        for k, v in six.iteritems(validation_threshold):
            if k.upper() == 'DSP':
                k = DeviceType.HEXAGON
            if k.upper() not in (DeviceType.CPU,
                                 DeviceType.GPU,
                                 DeviceType.HEXAGON,
                                 DeviceType.HTA,
                                 DeviceType.QUANTIZE):
                raise argparse.ArgumentTypeError(
                    'Unsupported validation threshold runtime: %s' % k)
            threshold_dict[k.upper()] = v
        model_config[YAMLKeyword.validation_threshold] = threshold_dict

        model_backend = model_config.get(YAMLKeyword.backend, "tensorflow")
        model_config[YAMLKeyword.backend] = model_backend
        validation_outputs_data = model_config.get(
            YAMLKeyword.validation_outputs_data, [])
        if not isinstance(validation_outputs_data, list):
            model_config[YAMLKeyword.validation_outputs_data] = [
                validation_outputs_data]
        else:
            model_config[YAMLKeyword.validation_outputs_data] = \
                validation_outputs_data

        for (subname, subgraph) in subgraphs.items():
            for key in [YAMLKeyword.input_tensors,
                        YAMLKeyword.input_shapes,
                        YAMLKeyword.output_tensors,
                        YAMLKeyword.output_shapes]:
                value = subgraph.get(key, "")
                mace_check(value != "", ModuleName.YAML_CONFIG,
                           "'%s' is necessary in %s" % (key, subname))
                if not isinstance(value, list):
                    subgraph[key] = [value]
                subgraph[key] = [str(v) for v in subgraph[key]]
                # --inputs_shapes will be passed to ELF file `mace_run_static',
                # if input_shapes contains spaces, such as: '1, 3, 224, 224',
                # because mace_run.cc use gflags to parse command line
                # arguments, --input_shapes 1, 3, 224, 224 will be passed as
                # `--input_shapes 1,'. So we strip out spaces here.
                if key in [YAMLKeyword.input_shapes,
                           YAMLKeyword.output_shapes]:
                    subgraph[key] = [e.replace(' ', '') for e in subgraph[key]]
            subgraph[YAMLKeyword.input_tensors] = \
                config_parser.to_list(subgraph[YAMLKeyword.input_tensors])
            input_size = len(subgraph[YAMLKeyword.input_tensors])
            subgraph[YAMLKeyword.output_tensors] = \
                config_parser.to_list(subgraph[YAMLKeyword.output_tensors])
            output_size = len(subgraph[YAMLKeyword.output_tensors])

            mace_check(len(subgraph[YAMLKeyword.input_shapes]) == input_size,
                       ModuleName.YAML_CONFIG,
                       "input shapes' size not equal inputs' size.")
            mace_check(len(subgraph[YAMLKeyword.output_shapes]) == output_size,
                       ModuleName.YAML_CONFIG,
                       "output shapes' size not equal outputs' size.")

            for key in [YAMLKeyword.input_data_types,
                        YAMLKeyword.output_data_types]:
                if key == YAMLKeyword.input_data_types:
                    count = input_size
                else:
                    count = output_size
                data_types = subgraph.get(key, "")
                if data_types:
                    if not isinstance(data_types, list):
                        subgraph[key] = [data_types] * count
                    for data_type in subgraph[key]:
                        mace_check(data_type in InOutDataTypeStrs,
                                   ModuleName.YAML_CONFIG,
                                   key + " must be in "
                                   + str(InOutDataTypeStrs))
                else:
                    subgraph[key] = [InOutDataType.float32] * count

            input_data_formats = \
                subgraph.get(YAMLKeyword.input_data_formats, [])
            if input_data_formats:
                if not isinstance(input_data_formats, list):
                    subgraph[YAMLKeyword.input_data_formats] = \
                        [input_data_formats] * input_size
                else:
                    mace_check(len(input_data_formats) == input_size,
                               ModuleName.YAML_CONFIG,
                               "input_data_formats should match"
                               " the size of input.")
                for input_data_format in \
                        subgraph[YAMLKeyword.input_data_formats]:
                    mace_check(input_data_format in DataFormatStrs,
                               ModuleName.YAML_CONFIG,
                               "'input_data_formats' must be in "
                               + str(DataFormatStrs) + ", but got "
                               + input_data_format)
            else:
                subgraph[YAMLKeyword.input_data_formats] = \
                    [DataFormat.NHWC] * input_size

            output_data_formats = subgraph.get(YAMLKeyword.output_data_formats,
                                               [])
            if output_data_formats:
                if not isinstance(output_data_formats, list):
                    subgraph[YAMLKeyword.output_data_formats] = \
                        [output_data_formats] * output_size
                else:
                    mace_check(len(output_data_formats)
                               == output_size,
                               ModuleName.YAML_CONFIG,
                               "output_data_formats should match"
                               " the size of output")
                for output_data_format in \
                        subgraph[YAMLKeyword.output_data_formats]:
                    mace_check(output_data_format in DataFormatStrs,
                               ModuleName.YAML_CONFIG,
                               "'output_data_formats' must be in "
                               + str(DataFormatStrs))
            else:
                subgraph[YAMLKeyword.output_data_formats] = \
                    [DataFormat.NHWC] * output_size

        for key in [YAMLKeyword.limit_opencl_kernel_time,
                    YAMLKeyword.opencl_queue_window_size,
                    YAMLKeyword.nnlib_graph_mode,
                    YAMLKeyword.obfuscate,
                    YAMLKeyword.winograd,
                    YAMLKeyword.quantize,
                    YAMLKeyword.quantize_large_weights,
                    YAMLKeyword.change_concat_ranges]:
            value = model_config.get(key, "")
            if value == "":
                model_config[key] = 0

        mace_check(model_config[YAMLKeyword.quantize] == 0 or
                   model_config[YAMLKeyword.quantize_large_weights] == 0,
                   ModuleName.YAML_CONFIG,
                   "quantize and quantize_large_weights should not be set to 1"
                   " at the same time.")

        mace_check(model_config[YAMLKeyword.winograd] in WinogradParameters,
                   ModuleName.YAML_CONFIG,
                   "'winograd' parameters must be in "
                   + str(WinogradParameters) +
                   ". 0 for disable winograd convolution")
        apu_version_with_no_device = \
            configs.get(YAMLKeyword.apu_version_with_no_device, "-1")
        configs[YAMLKeyword.apu_version_with_no_device] = \
            apu_version_with_no_device

    return configs


def clear_build_dirs(library_name):
    # make build dir
    if not os.path.exists(BUILD_OUTPUT_DIR):
        os.makedirs(BUILD_OUTPUT_DIR)
    # clear temp build dir
    tmp_build_dir = os.path.join(BUILD_OUTPUT_DIR, library_name,
                                 BUILD_TMP_DIR_NAME)
    if os.path.exists(tmp_build_dir):
        sh.rm('-rf', tmp_build_dir)
    os.makedirs(tmp_build_dir)
    # clear lib dir
    lib_output_dir = os.path.join(
        BUILD_OUTPUT_DIR, library_name, OUTPUT_LIBRARY_DIR_NAME)
    if os.path.exists(lib_output_dir):
        sh.rm('-rf', lib_output_dir)


################################
# convert
################################
def print_configuration(configs):
    title = "Common Configuration"
    header = ["key", "value"]
    data = list()
    data.append([YAMLKeyword.library_name,
                 configs[YAMLKeyword.library_name]])
    data.append([YAMLKeyword.target_abis,
                 configs[YAMLKeyword.target_abis]])
    data.append([YAMLKeyword.target_socs,
                 configs[YAMLKeyword.target_socs]])
    data.append([YAMLKeyword.model_graph_format,
                 configs[YAMLKeyword.model_graph_format]])
    data.append([YAMLKeyword.model_data_format,
                 configs[YAMLKeyword.model_data_format]])
    MaceLogger.summary(StringFormatter.table(header, data, title))


def build_model_lib(configs, address_sanitizer, debug_mode):
    MaceLogger.header(StringFormatter.block("Building model library"))

    # create model library dir
    library_name = configs[YAMLKeyword.library_name]
    for target_abi in configs[YAMLKeyword.target_abis]:
        model_lib_output_path = get_model_lib_output_path(library_name,
                                                          target_abi)
        library_out_dir = os.path.dirname(model_lib_output_path)
        if not os.path.exists(library_out_dir):
            os.makedirs(library_out_dir)
        toolchain = infer_toolchain(target_abi)
        sh_commands.bazel_build(
            MODEL_LIB_TARGET,
            abi=target_abi,
            toolchain=toolchain,
            enable_hexagon=hexagon_enabled(configs),
            enable_hta=hta_enabled(configs),
            enable_apu=apu_enabled(configs),
            enable_qnn=qnn_enabled(configs),
            enable_opencl=opencl_enabled(configs),
            enable_quantize=quantize_enabled(configs),
            enable_bfloat16=bfloat16_enabled(configs),
            enable_fp16=fp16_enabled(configs),
            address_sanitizer=address_sanitizer,
            symbol_hidden=get_symbol_hidden_mode(debug_mode),
            debug_mode=debug_mode
        )

        sh.cp("-f", MODEL_LIB_PATH, model_lib_output_path)


def print_library_summary(configs):
    library_name = configs[YAMLKeyword.library_name]
    title = "Library"
    header = ["key", "value"]
    data = list()
    data.append(["MACE Model Path",
                 "%s/%s/%s"
                 % (BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME)])
    if configs[YAMLKeyword.model_graph_format] == ModelFormat.code:
        data.append(["MACE Model Header Path",
                     "%s/%s/%s"
                     % (BUILD_OUTPUT_DIR, library_name,
                        MODEL_HEADER_DIR_PATH)])

    MaceLogger.summary(StringFormatter.table(header, data, title))


def convert_func(flags):
    configs = config_parser.parse(flags.config)

    print(configs)
    library_name = configs[YAMLKeyword.library_name]
    if not os.path.exists(BUILD_OUTPUT_DIR):
        os.makedirs(BUILD_OUTPUT_DIR)
    elif os.path.exists(os.path.join(BUILD_OUTPUT_DIR, library_name)):
        sh.rm("-rf", os.path.join(BUILD_OUTPUT_DIR, library_name))
    os.makedirs(os.path.join(BUILD_OUTPUT_DIR, library_name))
    if not os.path.exists(BUILD_DOWNLOADS_DIR):
        os.makedirs(BUILD_DOWNLOADS_DIR)

    model_output_dir = \
        '%s/%s/%s' % (BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME)
    model_header_dir = \
        '%s/%s/%s' % (BUILD_OUTPUT_DIR, library_name, MODEL_HEADER_DIR_PATH)
    # clear output dir
    if os.path.exists(model_output_dir):
        sh.rm("-rf", model_output_dir)
    os.makedirs(model_output_dir)
    if os.path.exists(model_header_dir):
        sh.rm("-rf", model_header_dir)

    if os.path.exists(MODEL_CODEGEN_DIR):
        sh.rm("-rf", MODEL_CODEGEN_DIR)
    if os.path.exists(ENGINE_CODEGEN_DIR):
        sh.rm("-rf", ENGINE_CODEGEN_DIR)

    if flags.quantize_stat:
        configs[YAMLKeyword.quantize_stat] = flags.quantize_stat

    if flags.model_data_format:
        model_data_format = flags.model_data_format
    else:
        model_data_format = configs.get(YAMLKeyword.model_data_format,
                                        "file")
    embed_model_data = model_data_format == ModelFormat.code

    if flags.model_graph_format:
        model_graph_format = flags.model_graph_format
    else:
        model_graph_format = configs.get(YAMLKeyword.model_graph_format,
                                         "file")
    embed_graph_def = model_graph_format == ModelFormat.code
    if flags.enable_micro:
        mace_check((not embed_model_data) and (not embed_graph_def),
                   ModuleName.YAML_CONFIG,
                   "You should specify file mode to convert micro model.")
    if embed_graph_def:
        os.makedirs(model_header_dir)
        sh_commands.gen_mace_engine_factory_source(
            configs[YAMLKeyword.models].keys(),
            embed_model_data)
        sh.cp("-f", glob.glob("mace/codegen/engine/*.h"),
              model_header_dir)

    convert.convert(configs, MODEL_CODEGEN_DIR, flags.enable_micro)

    for model_name, model_config in configs[YAMLKeyword.models].items():
        if flags.enable_micro:
            data_type = model_config.get(YAMLKeyword.data_type, "")
            if data_type != FPDataType.bf16_fp32.value:
                data_type = FPDataType.fp32_fp32.value

        model_codegen_dir = "%s/%s" % (MODEL_CODEGEN_DIR, model_name)
        encrypt.encrypt(model_name,
                        "%s/model/%s.pb" % (model_codegen_dir, model_name),
                        "%s/model/%s.data" % (model_codegen_dir, model_name),
                        model_codegen_dir,
                        bool(model_config.get(YAMLKeyword.obfuscate, 1)),
                        model_graph_format == "code",
                        model_data_format == "code")

        if model_graph_format == ModelFormat.file:
            sh.mv("-f",
                  '%s/model/%s.pb' % (model_codegen_dir, model_name),
                  model_output_dir)
            sh.mv("-f",
                  '%s/model/%s.data' % (model_codegen_dir, model_name),
                  model_output_dir)
            if flags.enable_micro:
                sh.mv("-f", '%s/model/%s_micro.tar.gz' %
                      (model_codegen_dir, model_name), model_output_dir)
        else:
            if not embed_model_data:
                sh.mv("-f",
                      '%s/model/%s.data' % (model_codegen_dir, model_name),
                      model_output_dir)

            sh.cp("-f", glob.glob("mace/codegen/models/*/code/*.h"),
                  model_header_dir)

        MaceLogger.summary(
            StringFormatter.block("Model %s converted" % model_name))

    if model_graph_format == ModelFormat.code:
        build_model_lib(configs, flags.address_sanitizer, flags.debug_mode)

    print_library_summary(configs)


################################
# run
################################
def build_mace_run(configs, target_abi, toolchain, address_sanitizer,
                   mace_lib_type, debug_mode, apu_version):
    library_name = configs[YAMLKeyword.library_name]

    build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
    if os.path.exists(build_tmp_binary_dir):
        sh.rm("-rf", build_tmp_binary_dir)
    os.makedirs(build_tmp_binary_dir)

    mace_run_target = MACE_RUN_STATIC_TARGET
    if mace_lib_type == MACELibType.dynamic:
        mace_run_target = MACE_RUN_DYNAMIC_TARGET
    build_arg = ""
    if configs[YAMLKeyword.model_graph_format] == ModelFormat.code:
        mace_check(os.path.exists(ENGINE_CODEGEN_DIR),
                   ModuleName.RUN,
                   "You should convert model first.")
        build_arg = "--per_file_copt=mace/tools/mace_run.cc@-DMODEL_GRAPH_FORMAT_CODE"  # noqa

    enable_apu = apu_enabled(configs)
    sh_commands.bazel_build(
        mace_run_target,
        abi=target_abi,
        toolchain=toolchain,
        enable_hexagon=hexagon_enabled(configs),
        enable_hta=hta_enabled(configs),
        enable_apu=enable_apu,
        apu_version=apu_version,
        enable_qnn=qnn_enabled(configs),
        enable_opencl=opencl_enabled(configs),
        enable_quantize=quantize_enabled(configs),
        enable_bfloat16=bfloat16_enabled(configs),
        enable_fp16=fp16_enabled(configs),
        address_sanitizer=address_sanitizer,
        symbol_hidden=get_symbol_hidden_mode(debug_mode, mace_lib_type),
        debug_mode=debug_mode,
        extra_args=build_arg
    )
    sh_commands.update_mace_run_binary(build_tmp_binary_dir,
                                       mace_lib_type == MACELibType.dynamic)


def print_package_summary(package_path):
    title = "Library"
    header = ["key", "value"]
    data = list()
    data.append(["MACE Model package Path",
                 package_path])

    MaceLogger.summary(StringFormatter.table(header, data, title))


def run_mace(flags):
    configs = format_model_config(flags)

    clear_build_dirs(configs[YAMLKeyword.library_name])

    target_socs = configs[YAMLKeyword.target_socs]
    device_list = DeviceManager.list_devices(flags.device_yml)
    if target_socs and TargetSOCTag.all not in target_socs:
        device_list = [dev for dev in device_list
                       if dev[YAMLKeyword.target_socs].lower() in target_socs]
    for target_abi in configs[YAMLKeyword.target_abis]:
        if flags.target_socs == TargetSOCTag.random:
            target_devices = sh_commands.choose_a_random_device(
                device_list, target_abi)
        else:
            target_devices = device_list
        # build target
        for dev in target_devices:
            if len(flags.devices_to_run) > 0 and \
                    dev['address'] not in flags.devices_to_run.split(','):
                continue
            if target_abi in dev[YAMLKeyword.target_abis]:
                # get toolchain
                toolchain = infer_toolchain(target_abi)

                device = DeviceWrapper(dev)
                apu_version = device.get_apu_version(apu_enabled(configs))
                build_mace_run(configs,
                               target_abi,
                               toolchain,
                               flags.address_sanitizer,
                               flags.mace_lib_type,
                               flags.debug_mode,
                               apu_version)
                # run
                start_time = time.time()
                with device.lock():
                    device.run_specify_abi(flags, configs, target_abi)
                elapse_minutes = (time.time() - start_time) / 60
                print("Elapse time: %f minutes." % elapse_minutes)
            elif dev[YAMLKeyword.device_name] != SystemType.host:
                six.print_('The device with soc %s do not support abi %s' %
                           (dev[YAMLKeyword.target_socs], target_abi),
                           file=sys.stderr)
            elif configs[YAMLKeyword.apu_version_with_no_device] != "-1":
                apu_version = configs[YAMLKeyword.apu_version_with_no_device]
                toolchain = infer_toolchain(target_abi)
                build_mace_run(configs,
                               target_abi,
                               toolchain,
                               flags.address_sanitizer,
                               flags.mace_lib_type,
                               flags.debug_mode,
                               apu_version)

    # package the output files
    package_path = sh_commands.packaging_lib(BUILD_OUTPUT_DIR,
                                             configs[YAMLKeyword.library_name])
    print_package_summary(package_path)


################################
# parsing arguments
################################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_to_caffe_env_type(v):
    if v.lower() == 'docker':
        return CaffeEnvType.DOCKER
    elif v.lower() == 'local':
        return CaffeEnvType.LOCAL
    else:
        raise argparse.ArgumentTypeError('[docker | local] expected.')


def str_to_mace_lib_type(v):
    if v.lower() == 'dynamic':
        return MACELibType.dynamic
    elif v.lower() == 'static':
        return MACELibType.static
    else:
        raise argparse.ArgumentTypeError('[dynamic| static] expected.')


def parse_args():
    """Parses command line arguments."""
    all_type_parent_parser = argparse.ArgumentParser(add_help=False)
    all_type_parent_parser.add_argument(
        '--devices_to_run',
        type=str,
        default="",
        help="devices you want to run")
    all_type_parent_parser.add_argument(
        '--config',
        type=str,
        default="",
        required=True,
        help="the path of model yaml configuration file.")
    all_type_parent_parser.add_argument(
        "--model_graph_format",
        type=str,
        default="",
        help="[file, code], MACE Model graph format.")
    all_type_parent_parser.add_argument(
        "--model_data_format",
        type=str,
        default="",
        help="['file', 'code'], MACE Model data format.")
    all_type_parent_parser.add_argument(
        "--target_abis",
        type=str,
        default="",
        help="Target ABIs, comma seperated list.")
    all_type_parent_parser.add_argument(
        "--target_socs",
        type=str,
        default="",
        help="Target SOCs, comma seperated list.")
    all_type_parent_parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Reserve debug symbols.")
    convert_run_parent_parser = argparse.ArgumentParser(add_help=False)
    convert_run_parent_parser.add_argument(
        '--address_sanitizer',
        action="store_true",
        help="Whether to use address sanitizer to check memory error")
    convert_run_parent_parser.add_argument(
        "--quantize_stat",
        action="store_true",
        help="whether to stat quantization range.")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    convert = subparsers.add_parser(
        'convert',
        parents=[all_type_parent_parser, convert_run_parent_parser],
        help='convert to mace model (file or code)')
    convert.add_argument(
        "--enable_micro",
        action="store_true",
        help="enable convert micro.")
    convert.set_defaults(func=convert_func)

    run = subparsers.add_parser(
        'run',
        parents=[all_type_parent_parser,
                 convert_run_parent_parser],
        help='run model in command line')
    run.set_defaults(func=run_mace)
    run.add_argument(
        "--mace_lib_type",
        type=str_to_mace_lib_type,
        default=DefaultValues.mace_lib_type,
        help="[static | dynamic], Which type MACE library to use.")
    run.add_argument(
        "--num_threads",
        type=int,
        default=DefaultValues.num_threads,
        help="num of threads")
    run.add_argument(
        "--cpu_affinity_policy",
        type=int,
        default=DefaultValues.cpu_affinity_policy,
        help="0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY")
    run.add_argument(
        "--gpu_perf_hint",
        type=int,
        default=DefaultValues.gpu_perf_hint,
        help="0:DEFAULT/1:LOW/2:NORMAL/3:HIGH")
    run.add_argument(
        "--gpu_priority_hint",
        type=int,
        default=DefaultValues.gpu_priority_hint,
        help="0:DEFAULT/1:LOW/2:NORMAL/3:HIGH")
    run.add_argument(
        "--device_yml",
        type=str,
        default='',
        help='embedded linux device config yml file'
    )
    run.add_argument(
        "--disable_tuning",
        action="store_true",
        help="Disable tuning for specific thread.")
    run.add_argument(
        "--round",
        type=int,
        default=1,
        help="The model running round.")
    run.add_argument(
        "--opencl_cache_reuse_policy",
        type=int,
        default=1,
        help="Policy used to reuse compatible opencl cache"
             "0:NONE/1:REUSE_SAME_GPU")
    run.add_argument(
        "--validate",
        action="store_true",
        help="whether to verify the results are consistent with "
             "the frameworks.")
    run.add_argument(
        "--layers",
        type=str,
        default="-1",
        help="'start_layer:end_layer' or 'layer', similar to python slice."
             " Use with --validate flag.")
    run.add_argument(
        "--caffe_env",
        type=str_to_caffe_env_type,
        default='docker',
        help="[docker | local] you can specific caffe environment for"
             " validation. local environment or caffe docker image.")
    run.add_argument(
        "--vlog_level",
        type=int,
        default=0,
        help="[1~5]. Verbose log level for debug.")
    run.add_argument(
        "--gpu_out_of_range_check",
        action="store_true",
        help="Enable out of memory check for gpu.")
    run.add_argument(
        "--restart_round",
        type=int,
        default=1,
        help="restart round between run.")
    run.add_argument(
        "--report",
        action="store_true",
        help="print run statistics report.")
    run.add_argument(
        "--report_dir",
        type=str,
        default="",
        help="print run statistics report.")
    run.add_argument(
        "--runtime_failure_ratio",
        type=float,
        default=0.0,
        help="[mock runtime failure ratio].")
    run.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="quantize stat input dir.")
    run.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="quantize stat output dir.")
    run.add_argument(
        "--cl_binary_to_code",
        action="store_true",
        help="convert OpenCL binaries to cpp.")
    run.add_argument(
        "--benchmark",
        action="store_true",
        help="enable op benchmark.")
    run.add_argument(
        "--fake_warmup",
        action="store_true",
        help="enable fake warmup.")
    run.add_argument(
        "--accelerator_cache_policy",
        type=int,
        default=DefaultValues.accelerator_cache_policy,
        help="0:NONE/1:STORE/2:LOAD/3:APU_LOAD_OR_STORE")
    run.add_argument(
        "--accelerator_binary_file",
        type=str,
        default="",
        help="accelerator cache load dir.")
    run.add_argument(
        "--accelerator_storage_file",
        type=str,
        default="",
        help="accelerator cache store dir.")
    run.add_argument(
        "--apu_boost_hint",
        type=int,
        default=DefaultValues.apu_boost_hint,
        help="The boost hint for APU frequency, ranged between 0 (lowest)"
             " to 100 (highest)")
    run.add_argument(
        "--apu_preference_hint",
        type=int,
        default=DefaultValues.apu_preference_hint,
        help="0:NEURON_PREFER_LOW_POWER"
             "1:NEURON_PREFER_FAST_SINGLE_ANSWER"
             "2:NEURON_PREFER_SUSTAINED_SPEED")
    run.add_argument(
        "--use_system_libhexagon_nn",
        action="store_true",
        help="use system libhexagon_nn_lib.so "
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    flags, unparsed = parse_args()
    flags.func(flags)
