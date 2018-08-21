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
import glob
import hashlib
import os
import re
import sh
import subprocess
import six
import sys
import urllib
import yaml

from enum import Enum

import sh_commands
from sh_commands import BuildType
from sh_commands import ModelFormat

from common import CaffeEnvType
from common import DeviceType
from common import mace_check
from common import MaceLogger
from common import StringFormatter

################################
# set environment
################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################
# common definitions
################################
BUILD_OUTPUT_DIR = 'builds'
BUILD_DOWNLOADS_DIR = BUILD_OUTPUT_DIR + '/downloads'
PHONE_DATA_DIR = "/data/local/tmp/mace_run"
MODEL_OUTPUT_DIR_NAME = 'model'
MODEL_HEADER_DIR_PATH = 'include/mace/public'
BUILD_TMP_DIR_NAME = '_tmp'
BUILD_TMP_GENERAL_OUTPUT_DIR_NAME = 'general'
OUTPUT_LIBRARY_DIR_NAME = 'lib'
OUTPUT_OPENCL_BINARY_DIR_NAME = 'opencl'
OUTPUT_OPENCL_BINARY_FILE_NAME = 'compiled_opencl_kernel'
OUTPUT_OPENCL_PARAMETER_FILE_NAME = 'tuned_opencl_parameter'
CL_COMPILED_BINARY_FILE_NAME = "mace_cl_compiled_program.bin"
CL_TUNED_PARAMETER_FILE_NAME = "mace_run.config"
CODEGEN_BASE_DIR = 'mace/codegen'
MODEL_CODEGEN_DIR = CODEGEN_BASE_DIR + '/models'
ENGINE_CODEGEN_DIR = CODEGEN_BASE_DIR + '/engine'
LIB_CODEGEN_DIR = CODEGEN_BASE_DIR + '/lib'
LIBMACE_SO_TARGET = "//mace/libmace:libmace.so"
LIBMACE_STATIC_TARGET = "//mace/libmace:libmace_static"
LIBMACE_STATIC_PATH = "bazel-genfiles/mace/libmace/libmace.a"
LIBMACE_DYNAMIC_PATH = "bazel-bin/mace/libmace/libmace.so"
MODEL_LIB_TARGET = "//mace/codegen:generated_models"
MODEL_LIB_PATH = "bazel-genfiles/mace/codegen/libgenerated_models.a"
MACE_RUN_STATIC_NAME = "mace_run_static"
MACE_RUN_DYNAMIC_NAME = "mace_run_dynamic"
MACE_RUN_STATIC_TARGET = "//mace/tools/validation:" + MACE_RUN_STATIC_NAME
MACE_RUN_DYNAMIC_TARGET = "//mace/tools/validation:" + MACE_RUN_DYNAMIC_NAME
QUANTIZE_STAT_TARGET = "//mace/tools/quantization:quantize_stat"
EXAMPLE_STATIC_NAME = "example_static"
EXAMPLE_DYNAMIC_NAME = "example_dynamic"
EXAMPLE_STATIC_TARGET = "//mace/examples/cli:" + EXAMPLE_STATIC_NAME
EXAMPLE_DYNAMIC_TARGET = "//mace/examples/cli:" + EXAMPLE_DYNAMIC_NAME
BM_MODEL_STATIC_NAME = "benchmark_model_static"
BM_MODEL_DYNAMIC_NAME = "benchmark_model_dynamic"
BM_MODEL_STATIC_TARGET = "//mace/benchmark:" + BM_MODEL_STATIC_NAME
BM_MODEL_DYNAMIC_TARGET = "//mace/benchmark:" + BM_MODEL_DYNAMIC_NAME
DEVICE_INTERIOR_DIR = PHONE_DATA_DIR + "/interior"
BUILD_TMP_OPENCL_BIN_DIR = 'opencl_bin'
ALL_SOC_TAG = 'all'

ABITypeStrs = [
    'armeabi-v7a',
    'arm64-v8a',
    'host',
]


class ABIType(object):
    armeabi_v7a = 'armeabi-v7a'
    arm64_v8a = 'arm64-v8a'
    host = 'host'


ModelFormatStrs = [
    "file",
    "code",
]


class MACELibType(object):
    static = 0
    dynamic = 1


PlatformTypeStrs = [
    "tensorflow",
    "caffe",
]
PlatformType = Enum('PlatformType', [(ele, ele) for ele in PlatformTypeStrs],
                    type=str)

RuntimeTypeStrs = [
    "cpu",
    "gpu",
    "dsp",
    "cpu+gpu"
]


class RuntimeType(object):
    cpu = 'cpu'
    gpu = 'gpu'
    dsp = 'dsp'
    cpu_gpu = 'cpu+gpu'


InputDataTypeStrs = [
    "int32",
    "float32",
]

InputDataType = Enum('InputDataType',
                     [(ele, ele) for ele in InputDataTypeStrs],
                     type=str)


CPUDataTypeStrs = [
    "fp32",
]

CPUDataType = Enum('CPUDataType', [(ele, ele) for ele in CPUDataTypeStrs],
                   type=str)

GPUDataTypeStrs = [
    "fp16_fp32",
    "fp32_fp32",
]

GPUDataType = Enum('GPUDataType', [(ele, ele) for ele in GPUDataTypeStrs],
                   type=str)

DSPDataTypeStrs = [
    "uint8",
]

DSPDataType = Enum('DSPDataType', [(ele, ele) for ele in DSPDataTypeStrs],
                   type=str)

WinogradParameters = [0, 2, 4]


class DefaultValues(object):
    mace_lib_type = MACELibType.static
    omp_num_threads = -1,
    cpu_affinity_policy = 1,
    gpu_perf_hint = 3,
    gpu_priority_hint = 3,


class YAMLKeyword(object):
    library_name = 'library_name'
    target_abis = 'target_abis'
    target_socs = 'target_socs'
    model_graph_format = 'model_graph_format'
    model_data_format = 'model_data_format'
    models = 'models'
    platform = 'platform'
    model_file_path = 'model_file_path'
    model_sha256_checksum = 'model_sha256_checksum'
    weight_file_path = 'weight_file_path'
    weight_sha256_checksum = 'weight_sha256_checksum'
    subgraphs = 'subgraphs'
    input_tensors = 'input_tensors'
    input_shapes = 'input_shapes'
    input_ranges = 'input_ranges'
    output_tensors = 'output_tensors'
    output_shapes = 'output_shapes'
    runtime = 'runtime'
    data_type = 'data_type'
    input_data_types = 'input_data_types'
    limit_opencl_kernel_time = 'limit_opencl_kernel_time'
    nnlib_graph_mode = 'nnlib_graph_mode'
    obfuscate = 'obfuscate'
    winograd = 'winograd'
    quantize = 'quantize'
    quantize_range_file = 'quantize_range_file'
    validation_inputs_data = 'validation_inputs_data'
    validation_threshold = 'validation_threshold'
    graph_optimize_options = 'graph_optimize_options'  # internal use for now


class ModuleName(object):
    YAML_CONFIG = 'YAML CONFIG'
    MODEL_CONVERTER = 'Model Converter'
    RUN = 'RUN'
    BENCHMARK = 'Benchmark'


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
    elif runtime == RuntimeType.gpu:
        device_type = DeviceType.GPU
    elif runtime == RuntimeType.cpu:
        device_type = DeviceType.CPU

    return device_type


def get_hexagon_mode(configs):
    runtime_list = []
    for model_name in configs[YAMLKeyword.models]:
        model_runtime =\
            configs[YAMLKeyword.models][model_name].get(
                YAMLKeyword.runtime, "")
        runtime_list.append(model_runtime.lower())

    if RuntimeType.dsp in runtime_list:
        return True
    return False


def get_opencl_mode(configs):
    runtime_list = []
    for model_name in configs[YAMLKeyword.models]:
        model_runtime =\
            configs[YAMLKeyword.models][model_name].get(
                YAMLKeyword.runtime, "")
        runtime_list.append(model_runtime.lower())

    if RuntimeType.gpu in runtime_list or RuntimeType.cpu_gpu in runtime_list:
        return True
    return False


def md5sum(str):
    md5 = hashlib.md5()
    md5.update(str)
    return md5.hexdigest()


def sha256_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


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
    if flags.target_socs:
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
        if ALL_SOC_TAG in target_socs:
            mace_check(available_socs,
                       ModuleName.YAML_CONFIG,
                       "Build for all SOCs plugged in computer, "
                       "you at least plug in one phone")
        else:
            for soc in target_socs:
                mace_check(soc in available_socs,
                           ModuleName.YAML_CONFIG,
                           "Build specified SOC library, "
                           "you must plug in a phone using the SOC")

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
            weight_checksum =\
                model_config.get(YAMLKeyword.weight_sha256_checksum, "")
            mace_check(weight_checksum != "", ModuleName.YAML_CONFIG,
                       "'%s' is necessary" %
                       YAMLKeyword.weight_sha256_checksum)
        else:
            model_config[YAMLKeyword.weight_sha256_checksum] = ""

        runtime = model_config.get(YAMLKeyword.runtime, "")
        mace_check(runtime in RuntimeTypeStrs,
                   ModuleName.YAML_CONFIG,
                   "'runtime' must be in " + str(RuntimeTypeStrs))
        if ABIType.host in target_abis:
            mace_check(runtime == RuntimeType.cpu,
                       ModuleName.YAML_CONFIG,
                       "host only support cpu runtime now.")

        data_type = model_config.get(YAMLKeyword.data_type, "")
        if runtime == RuntimeType.cpu_gpu and data_type not in GPUDataTypeStrs:
            model_config[YAMLKeyword.data_type] = \
                GPUDataType.fp16_fp32.value
        elif runtime == RuntimeType.cpu:
            if len(data_type) > 0:
                mace_check(data_type in CPUDataTypeStrs,
                           ModuleName.YAML_CONFIG,
                           "'data_type' must be in " + str(CPUDataTypeStrs)
                           + " for cpu runtime")
            else:
                model_config[YAMLKeyword.data_type] = \
                    CPUDataType.fp32.value
        elif runtime == RuntimeType.gpu:
            if len(data_type) > 0:
                mace_check(data_type in GPUDataTypeStrs,
                           ModuleName.YAML_CONFIG,
                           "'data_type' must be in " + str(GPUDataTypeStrs)
                           + " for gpu runtime")
            else:
                model_config[YAMLKeyword.data_type] =\
                    GPUDataType.fp16_fp32.value
        elif runtime == RuntimeType.dsp:
            if len(data_type) > 0:
                mace_check(data_type in DSPDataTypeStrs,
                           ModuleName.YAML_CONFIG,
                           "'data_type' must be in " + str(DSPDataTypeStrs)
                           + " for dsp runtime")
            else:
                model_config[YAMLKeyword.data_type] = \
                    DSPDataType.uint8.value

        subgraphs = model_config.get(YAMLKeyword.subgraphs, "")
        mace_check(len(subgraphs) > 0, ModuleName.YAML_CONFIG,
                   "at least one subgraph is needed")

        for subgraph in subgraphs:
            for key in [YAMLKeyword.input_tensors,
                        YAMLKeyword.input_shapes,
                        YAMLKeyword.output_tensors,
                        YAMLKeyword.output_shapes]:
                value = subgraph.get(key, "")
                mace_check(value != "", ModuleName.YAML_CONFIG,
                           "'%s' is necessary in subgraph" % key)
                if not isinstance(value, list):
                    subgraph[key] = [value]
                subgraph[key] = [str(v) for v in subgraph[key]]

            input_data_types = subgraph.get(YAMLKeyword.input_data_types, "")
            if input_data_types:
                if not isinstance(input_data_types, list):
                    subgraph[YAMLKeyword.input_data_types] = [input_data_types]
                for input_data_type in input_data_types:
                    mace_check(input_data_type in InputDataTypeStrs,
                               ModuleName.YAML_CONFIG,
                               "'input_data_types' must be in "
                               + str(InputDataTypeStrs))
            else:
                subgraph[YAMLKeyword.input_data_types] = []

            validation_threshold = subgraph.get(
                YAMLKeyword.validation_threshold, {})
            if not isinstance(validation_threshold, dict):
                raise argparse.ArgumentTypeError(
                        'similarity threshold must be a dict.')

            threshold_dict = {
                    DeviceType.CPU: 0.999,
                    DeviceType.GPU: 0.995,
                    DeviceType.HEXAGON: 0.930,
                    }
            for k, v in six.iteritems(validation_threshold):
                if k.upper() == 'DSP':
                    k = DeviceType.HEXAGON
                if k.upper() not in (DeviceType.CPU,
                                     DeviceType.GPU,
                                     DeviceType.HEXAGON):
                    raise argparse.ArgumentTypeError(
                            'Unsupported validation threshold runtime: %s' % k)
                threshold_dict[k.upper()] = v

            subgraph[YAMLKeyword.validation_threshold] = threshold_dict

            validation_inputs_data = subgraph.get(
                YAMLKeyword.validation_inputs_data, [])
            if not isinstance(validation_inputs_data, list):
                subgraph[YAMLKeyword.validation_inputs_data] = [
                    validation_inputs_data]
            else:
                subgraph[YAMLKeyword.validation_inputs_data] = \
                    validation_inputs_data
            input_ranges = subgraph.get(
                YAMLKeyword.input_ranges, [])
            if not isinstance(input_ranges, list):
                subgraph[YAMLKeyword.input_ranges] = [input_ranges]
            else:
                subgraph[YAMLKeyword.input_ranges] = input_ranges
            subgraph[YAMLKeyword.input_ranges] =\
                [str(v) for v in subgraph[YAMLKeyword.input_ranges]]

        for key in [YAMLKeyword.limit_opencl_kernel_time,
                    YAMLKeyword.nnlib_graph_mode,
                    YAMLKeyword.obfuscate,
                    YAMLKeyword.winograd,
                    YAMLKeyword.quantize]:
            value = model_config.get(key, "")
            if value == "":
                model_config[key] = 0

        mace_check(model_config[YAMLKeyword.winograd] in WinogradParameters,
                   ModuleName.YAML_CONFIG,
                   "'winograd' parameters must be in "
                   + str(WinogradParameters) +
                   ". 0 for disable winograd convolution")

        weight_file_path = model_config.get(YAMLKeyword.weight_file_path, "")
        model_config[YAMLKeyword.weight_file_path] = weight_file_path

    return configs


def get_build_binary_dir(library_name, target_abi):
    return "%s/%s/%s/%s" % (
        BUILD_OUTPUT_DIR, library_name, BUILD_TMP_DIR_NAME, target_abi)


def get_build_model_dirs(library_name, model_name, target_abi, target_soc,
                         serial_num, model_file_path):
    model_path_digest = md5sum(model_file_path)
    model_output_base_dir = "%s/%s/%s/%s/%s" % (
        BUILD_OUTPUT_DIR, library_name, BUILD_TMP_DIR_NAME,
        model_name, model_path_digest)

    if target_abi == ABIType.host:
        model_output_dir = "%s/%s" % (model_output_base_dir, target_abi)
    elif not target_soc or not serial_num:
        model_output_dir = "%s/%s/%s" % (
            model_output_base_dir, BUILD_TMP_GENERAL_OUTPUT_DIR_NAME,
            target_abi)
    else:
        device_name = \
            sh_commands.adb_get_device_name_by_serialno(serial_num)
        model_output_dir = "%s/%s_%s/%s" % (
            model_output_base_dir, device_name,
            target_soc, target_abi)

    mace_model_dir = \
        '%s/%s/%s' % (BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME)

    return model_output_base_dir, model_output_dir, mace_model_dir


def get_opencl_binary_output_path(library_name, target_abi,
                                  target_soc, serial_num):
    device_name = \
        sh_commands.adb_get_device_name_by_serialno(serial_num)
    return '%s/%s/%s/%s/%s_%s.%s.%s.bin' % \
           (BUILD_OUTPUT_DIR,
            library_name,
            OUTPUT_OPENCL_BINARY_DIR_NAME,
            target_abi,
            library_name,
            OUTPUT_OPENCL_BINARY_FILE_NAME,
            device_name,
            target_soc)


def get_opencl_parameter_output_path(library_name, target_abi,
                                     target_soc, serial_num):
    device_name = \
        sh_commands.adb_get_device_name_by_serialno(serial_num)
    return '%s/%s/%s/%s/%s_%s.%s.%s.bin' % \
           (BUILD_OUTPUT_DIR,
            library_name,
            OUTPUT_OPENCL_BINARY_DIR_NAME,
            target_abi,
            library_name,
            OUTPUT_OPENCL_PARAMETER_FILE_NAME,
            device_name,
            target_soc)


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


def check_model_converted(library_name, model_name,
                          model_graph_format, model_data_format,
                          abi):
    model_output_dir = \
        '%s/%s/%s' % (BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME)
    if model_graph_format == ModelFormat.file:
        mace_check(os.path.exists("%s/%s.pb" % (model_output_dir, model_name)),
                   ModuleName.RUN,
                   "You should convert model first.")
    else:
        model_lib_path = get_model_lib_output_path(library_name, abi)
        mace_check(os.path.exists(model_lib_path),
                   ModuleName.RUN,
                   "You should convert model first.")
    if model_data_format == ModelFormat.file:
        mace_check(os.path.exists("%s/%s.data" %
                                  (model_output_dir, model_name)),
                   ModuleName.RUN,
                   "You should convert model first.")


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


def get_model_files(model_file_path,
                    model_sha256_checksum,
                    model_output_dir,
                    weight_file_path="",
                    weight_sha256_checksum=""):
    model_file = model_file_path
    weight_file = weight_file_path

    if model_file_path.startswith("http://") or \
            model_file_path.startswith("https://"):
        model_file = model_output_dir + "/" + md5sum(model_file_path) + ".pb"
        if not os.path.exists(model_file) or \
                sha256_checksum(model_file) != model_sha256_checksum:
            MaceLogger.info("Downloading model, please wait ...")
            urllib.urlretrieve(model_file_path, model_file)
            MaceLogger.info("Model downloaded successfully.")

    if sha256_checksum(model_file) != model_sha256_checksum:
        MaceLogger.error(ModuleName.MODEL_CONVERTER,
                         "model file sha256checksum not match")

    if weight_file_path.startswith("http://") or \
            weight_file_path.startswith("https://"):
        weight_file = \
            model_output_dir + "/" + md5sum(weight_file_path) + ".caffemodel"
        if not os.path.exists(weight_file) or \
                sha256_checksum(weight_file) != weight_sha256_checksum:
            MaceLogger.info("Downloading model weight, please wait ...")
            urllib.urlretrieve(weight_file_path, weight_file)
            MaceLogger.info("Model weight downloaded successfully.")

    if weight_file:
        if sha256_checksum(weight_file) != weight_sha256_checksum:
            MaceLogger.error(ModuleName.MODEL_CONVERTER,
                             "weight file sha256checksum not match")

    return model_file, weight_file


def convert_model(configs):
    # Remove previous output dirs
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

    embed_model_data = \
        configs[YAMLKeyword.model_data_format] == ModelFormat.code

    if os.path.exists(MODEL_CODEGEN_DIR):
        sh.rm("-rf", MODEL_CODEGEN_DIR)
    if os.path.exists(ENGINE_CODEGEN_DIR):
        sh.rm("-rf", ENGINE_CODEGEN_DIR)

    if configs[YAMLKeyword.model_graph_format] == ModelFormat.code:
        os.makedirs(model_header_dir)
        sh_commands.gen_mace_engine_factory_source(
            configs[YAMLKeyword.models].keys(),
            embed_model_data)
        sh.cp("-f", glob.glob("mace/codegen/engine/*.h"),
              model_header_dir)

    for model_name in configs[YAMLKeyword.models]:
        MaceLogger.header(
            StringFormatter.block("Convert %s model" % model_name))
        model_config = configs[YAMLKeyword.models][model_name]
        runtime = model_config[YAMLKeyword.runtime]

        model_file_path, weight_file_path = get_model_files(
            model_config[YAMLKeyword.model_file_path],
            model_config[YAMLKeyword.model_sha256_checksum],
            BUILD_DOWNLOADS_DIR,
            model_config[YAMLKeyword.weight_file_path],
            model_config[YAMLKeyword.weight_sha256_checksum])

        data_type = model_config[YAMLKeyword.data_type]
        # TODO(liuqi): support multiple subgraphs
        subgraphs = model_config[YAMLKeyword.subgraphs]

        model_codegen_dir = "%s/%s" % (MODEL_CODEGEN_DIR, model_name)
        sh_commands.gen_model_code(
            model_codegen_dir,
            model_config[YAMLKeyword.platform],
            model_file_path,
            weight_file_path,
            model_config[YAMLKeyword.model_sha256_checksum],
            model_config[YAMLKeyword.weight_sha256_checksum],
            ",".join(subgraphs[0][YAMLKeyword.input_tensors]),
            ",".join(subgraphs[0][YAMLKeyword.output_tensors]),
            runtime,
            model_name,
            ":".join(subgraphs[0][YAMLKeyword.input_shapes]),
            model_config[YAMLKeyword.nnlib_graph_mode],
            embed_model_data,
            model_config[YAMLKeyword.winograd],
            model_config[YAMLKeyword.quantize],
            model_config.get(YAMLKeyword.quantize_range_file, ""),
            model_config[YAMLKeyword.obfuscate],
            configs[YAMLKeyword.model_graph_format],
            data_type,
            ",".join(model_config.get(YAMLKeyword.graph_optimize_options, [])))

        if configs[YAMLKeyword.model_graph_format] == ModelFormat.file:
            sh.mv("-f",
                  '%s/%s.pb' % (model_codegen_dir, model_name),
                  model_output_dir)
            sh.mv("-f",
                  '%s/%s.data' % (model_codegen_dir, model_name),
                  model_output_dir)
        else:
            if not embed_model_data:
                sh.mv("-f",
                      '%s/%s.data' % (model_codegen_dir, model_name),
                      model_output_dir)
            sh.cp("-f", glob.glob("mace/codegen/models/*/*.h"),
                  model_header_dir)

        MaceLogger.summary(
            StringFormatter.block("Model %s converted" % model_name))


def get_model_lib_output_path(library_name, abi):
    lib_output_path = os.path.join(BUILD_OUTPUT_DIR, library_name,
                                   MODEL_OUTPUT_DIR_NAME, abi,
                                   "%s.a" % library_name)
    return lib_output_path


def build_model_lib(configs, address_sanitizer):
    MaceLogger.header(StringFormatter.block("Building model library"))

    # create model library dir
    library_name = configs[YAMLKeyword.library_name]
    for target_abi in configs[YAMLKeyword.target_abis]:
        hexagon_mode = get_hexagon_mode(configs)
        model_lib_output_path = get_model_lib_output_path(library_name,
                                                          target_abi)
        library_out_dir = os.path.dirname(model_lib_output_path)
        if not os.path.exists(library_out_dir):
            os.makedirs(library_out_dir)

        sh_commands.bazel_build(
            MODEL_LIB_TARGET,
            abi=target_abi,
            hexagon_mode=hexagon_mode,
            enable_opencl=get_opencl_mode(configs),
            address_sanitizer=address_sanitizer,
            symbol_hidden=True
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
    configs = format_model_config(flags)

    print_configuration(configs)

    convert_model(configs)

    if configs[YAMLKeyword.model_graph_format] == ModelFormat.code:
        build_model_lib(configs, flags.address_sanitizer)

    print_library_summary(configs)


################################
# run
################################
def report_run_statistics(stdout,
                          abi,
                          serialno,
                          model_name,
                          device_type,
                          output_dir,
                          tuned):
    metrics = [0] * 3
    for line in stdout.split('\n'):
        line = line.strip()
        parts = line.split()
        if len(parts) == 4 and parts[0].startswith("time"):
            metrics[0] = str(float(parts[1]))
            metrics[1] = str(float(parts[2]))
            metrics[2] = str(float(parts[3]))
            break

    device_name = ""
    target_soc = ""
    if abi != "host":
        props = sh_commands.adb_getprop_by_serialno(serialno)
        device_name = props.get("ro.product.model", "")
        target_soc = props.get("ro.board.platform", "")

    report_filename = output_dir + "/report.csv"
    if not os.path.exists(report_filename):
        with open(report_filename, 'w') as f:
            f.write("model_name,device_name,soc,abi,runtime,"
                    "init(ms),warmup(ms),run_avg(ms),tuned\n")

    data_str = "{model_name},{device_name},{soc},{abi},{device_type}," \
               "{init},{warmup},{run_avg},{tuned}\n" \
        .format(model_name=model_name,
                device_name=device_name,
                soc=target_soc,
                abi=abi,
                device_type=device_type,
                init=metrics[0],
                warmup=metrics[1],
                run_avg=metrics[2],
                tuned=tuned)
    with open(report_filename, 'a') as f:
        f.write(data_str)


def build_mace_run(configs, target_abi, enable_openmp, address_sanitizer,
                   mace_lib_type):
    library_name = configs[YAMLKeyword.library_name]
    hexagon_mode = get_hexagon_mode(configs)

    build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
    if os.path.exists(build_tmp_binary_dir):
        sh.rm("-rf", build_tmp_binary_dir)
    os.makedirs(build_tmp_binary_dir)

    symbol_hidden = True
    mace_run_target = MACE_RUN_STATIC_TARGET
    if mace_lib_type == MACELibType.dynamic:
        symbol_hidden = False
        mace_run_target = MACE_RUN_DYNAMIC_TARGET
    build_arg = ""
    if configs[YAMLKeyword.model_graph_format] == ModelFormat.code:
        mace_check(os.path.exists(ENGINE_CODEGEN_DIR),
                   ModuleName.RUN,
                   "You should convert model first.")
        build_arg = "--per_file_copt=mace/tools/validation/mace_run.cc@-DMODEL_GRAPH_FORMAT_CODE"  # noqa

    sh_commands.bazel_build(
        mace_run_target,
        abi=target_abi,
        hexagon_mode=hexagon_mode,
        enable_openmp=enable_openmp,
        enable_opencl=get_opencl_mode(configs),
        address_sanitizer=address_sanitizer,
        symbol_hidden=symbol_hidden,
        extra_args=build_arg
    )
    sh_commands.update_mace_run_binary(build_tmp_binary_dir,
                                       mace_lib_type == MACELibType.dynamic)


def build_quantize_stat(configs):
    library_name = configs[YAMLKeyword.library_name]

    build_tmp_binary_dir = get_build_binary_dir(library_name, ABIType.host)
    if os.path.exists(build_tmp_binary_dir):
        sh.rm("-rf", build_tmp_binary_dir)
    os.makedirs(build_tmp_binary_dir)

    quantize_stat_target = QUANTIZE_STAT_TARGET
    build_arg = ""
    print (configs[YAMLKeyword.model_graph_format])
    if configs[YAMLKeyword.model_graph_format] == ModelFormat.code:
        mace_check(os.path.exists(ENGINE_CODEGEN_DIR),
                   ModuleName.RUN,
                   "You should convert model first.")
        build_arg = "--per_file_copt=mace/tools/quantization/quantize_stat.cc@-DMODEL_GRAPH_FORMAT_CODE"  # noqa

    sh_commands.bazel_build(
        quantize_stat_target,
        abi=ABIType.host,
        enable_openmp=True,
        symbol_hidden=True,
        extra_args=build_arg
    )

    quantize_stat_filepath = build_tmp_binary_dir + "/quantize_stat"
    if os.path.exists(quantize_stat_filepath):
        sh.rm("-rf", quantize_stat_filepath)
    sh.cp("-f", "bazel-bin/mace/tools/quantization/quantize_stat",
          build_tmp_binary_dir)


def build_example(configs, target_abi, enable_openmp, mace_lib_type):
    library_name = configs[YAMLKeyword.library_name]
    hexagon_mode = get_hexagon_mode(configs)

    build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
    if os.path.exists(build_tmp_binary_dir):
        sh.rm("-rf", build_tmp_binary_dir)
    os.makedirs(build_tmp_binary_dir)

    symbol_hidden = True
    libmace_target = LIBMACE_STATIC_TARGET
    if mace_lib_type == MACELibType.dynamic:
        symbol_hidden = False
        libmace_target = LIBMACE_SO_TARGET

    sh_commands.bazel_build(libmace_target,
                            abi=target_abi,
                            enable_openmp=enable_openmp,
                            enable_opencl=get_opencl_mode(configs),
                            hexagon_mode=hexagon_mode,
                            symbol_hidden=symbol_hidden)

    if os.path.exists(LIB_CODEGEN_DIR):
        sh.rm("-rf", LIB_CODEGEN_DIR)
    sh.mkdir("-p", LIB_CODEGEN_DIR)

    build_arg = ""
    if configs[YAMLKeyword.model_graph_format] == ModelFormat.code:
        mace_check(os.path.exists(ENGINE_CODEGEN_DIR),
                   ModuleName.RUN,
                   "You should convert model first.")
        model_lib_path = get_model_lib_output_path(library_name,
                                                   target_abi)
        sh.cp("-f", model_lib_path, LIB_CODEGEN_DIR)
        build_arg = "--per_file_copt=mace/examples/cli/example.cc@-DMODEL_GRAPH_FORMAT_CODE"  # noqa

    if mace_lib_type == MACELibType.dynamic:
        example_target = EXAMPLE_DYNAMIC_TARGET
        sh.cp("-f", LIBMACE_DYNAMIC_PATH, LIB_CODEGEN_DIR)
    else:
        example_target = EXAMPLE_STATIC_TARGET
        sh.cp("-f", LIBMACE_STATIC_PATH, LIB_CODEGEN_DIR)

    sh_commands.bazel_build(example_target,
                            abi=target_abi,
                            enable_openmp=enable_openmp,
                            enable_opencl=get_opencl_mode(configs),
                            hexagon_mode=hexagon_mode,
                            extra_args=build_arg)

    target_bin = "/".join(sh_commands.bazel_target_to_bin(example_target))
    sh.cp("-f", target_bin, build_tmp_binary_dir)
    if os.path.exists(LIB_CODEGEN_DIR):
        sh.rm("-rf", LIB_CODEGEN_DIR)


def tuning(library_name, model_name, model_config,
           model_graph_format, model_data_format,
           target_abi, target_soc, serial_num,
           mace_lib_type):
    print('* Tuning, it may take some time...')

    build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
    mace_run_name = MACE_RUN_STATIC_NAME
    link_dynamic = False
    if mace_lib_type == MACELibType.dynamic:
        mace_run_name = MACE_RUN_DYNAMIC_NAME
        link_dynamic = True

    embed_model_data = model_data_format == ModelFormat.code

    model_output_base_dir, model_output_dir, mace_model_dir = \
        get_build_model_dirs(library_name, model_name, target_abi,
                             target_soc, serial_num,
                             model_config[YAMLKeyword.model_file_path])

    # build for specified soc
    sh_commands.clear_phone_data_dir(serial_num, PHONE_DATA_DIR)

    subgraphs = model_config[YAMLKeyword.subgraphs]
    # generate input data
    sh_commands.gen_random_input(
        model_output_dir,
        subgraphs[0][YAMLKeyword.input_tensors],
        subgraphs[0][YAMLKeyword.input_shapes],
        subgraphs[0][YAMLKeyword.validation_inputs_data],
        input_ranges=subgraphs[0][YAMLKeyword.input_ranges],
        input_data_types=subgraphs[0][YAMLKeyword.input_data_types])

    sh_commands.tuning_run(
        abi=target_abi,
        serialno=serial_num,
        target_dir=build_tmp_binary_dir,
        target_name=mace_run_name,
        vlog_level=0,
        embed_model_data=embed_model_data,
        model_output_dir=model_output_dir,
        input_nodes=subgraphs[0][YAMLKeyword.input_tensors],
        output_nodes=subgraphs[0][YAMLKeyword.output_tensors],
        input_shapes=subgraphs[0][YAMLKeyword.input_shapes],
        output_shapes=subgraphs[0][YAMLKeyword.output_shapes],
        mace_model_dir=mace_model_dir,
        model_tag=model_name,
        device_type=DeviceType.GPU,
        running_round=0,
        restart_round=1,
        limit_opencl_kernel_time=model_config[YAMLKeyword.limit_opencl_kernel_time],  # noqa
        tuning=True,
        out_of_range_check=False,
        phone_data_dir=PHONE_DATA_DIR,
        model_graph_format=model_graph_format,
        opencl_binary_file="",
        opencl_parameter_file="",
        libmace_dynamic_library_path=LIBMACE_DYNAMIC_PATH,
        link_dynamic=link_dynamic,
    )
    # pull opencl binary
    sh_commands.pull_file_from_device(
        serial_num,
        DEVICE_INTERIOR_DIR,
        CL_COMPILED_BINARY_FILE_NAME,
        "%s/%s" % (model_output_dir, BUILD_TMP_OPENCL_BIN_DIR))

    # pull opencl parameter
    sh_commands.pull_file_from_device(
        serial_num,
        PHONE_DATA_DIR,
        CL_TUNED_PARAMETER_FILE_NAME,
        "%s/%s" % (model_output_dir, BUILD_TMP_OPENCL_BIN_DIR))

    print('Tuning done\n')


def run_specific_target(flags, configs, target_abi,
                        target_soc, serial_num):
    library_name = configs[YAMLKeyword.library_name]
    mace_lib_type = flags.mace_lib_type
    embed_model_data = \
        configs[YAMLKeyword.model_data_format] == ModelFormat.code
    build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)

    # get target name for run
    if flags.example:
        if mace_lib_type == MACELibType.static:
            target_name = EXAMPLE_STATIC_NAME
        else:
            target_name = EXAMPLE_DYNAMIC_NAME
    else:
        if mace_lib_type == MACELibType.static:
            target_name = MACE_RUN_STATIC_NAME
        else:
            target_name = MACE_RUN_DYNAMIC_NAME

    link_dynamic = mace_lib_type == MACELibType.dynamic
    model_output_dirs = []

    for model_name in configs[YAMLKeyword.models]:
        check_model_converted(library_name, model_name,
                              configs[YAMLKeyword.model_graph_format],
                              configs[YAMLKeyword.model_data_format],
                              target_abi)
        if target_abi == ABIType.host:
            device_name = ABIType.host
        else:
            device_name = \
                sh_commands.adb_get_device_name_by_serialno(serial_num)
            sh_commands.clear_phone_data_dir(serial_num, PHONE_DATA_DIR)

        MaceLogger.header(
            StringFormatter.block(
                "Run model %s on %s" % (model_name, device_name)))

        model_config = configs[YAMLKeyword.models][model_name]
        model_runtime = model_config[YAMLKeyword.runtime]
        subgraphs = model_config[YAMLKeyword.subgraphs]

        if not configs[YAMLKeyword.target_socs] or target_abi == ABIType.host:
            model_output_base_dir, model_output_dir, mace_model_dir = \
                get_build_model_dirs(library_name, model_name, target_abi,
                                     None, None,
                                     model_config[YAMLKeyword.model_file_path])
        else:
            model_output_base_dir, model_output_dir, mace_model_dir = \
                get_build_model_dirs(library_name, model_name, target_abi,
                                     target_soc, serial_num,
                                     model_config[YAMLKeyword.model_file_path])
        # clear temp model output dir
        if os.path.exists(model_output_dir):
            sh.rm("-rf", model_output_dir)
        os.makedirs(model_output_dir)

        is_tuned = False
        model_opencl_output_bin_path = ""
        model_opencl_parameter_path = ""
        # tuning for specified soc
        if not flags.address_sanitizer \
                and not flags.example \
                and target_abi != ABIType.host \
                and configs[YAMLKeyword.target_socs] \
                and target_soc \
                and model_runtime in [RuntimeType.gpu, RuntimeType.cpu_gpu] \
                and not flags.disable_tuning:
            tuning(library_name, model_name, model_config,
                   configs[YAMLKeyword.model_graph_format],
                   configs[YAMLKeyword.model_data_format],
                   target_abi, target_soc, serial_num,
                   mace_lib_type)
            model_output_dirs.append(model_output_dir)
            model_opencl_output_bin_path =\
                "%s/%s/%s" % (model_output_dir,
                              BUILD_TMP_OPENCL_BIN_DIR,
                              CL_COMPILED_BINARY_FILE_NAME)
            model_opencl_parameter_path = \
                "%s/%s/%s" % (model_output_dir,
                              BUILD_TMP_OPENCL_BIN_DIR,
                              CL_TUNED_PARAMETER_FILE_NAME)
            sh_commands.clear_phone_data_dir(serial_num, PHONE_DATA_DIR)
            is_tuned = True
        elif target_abi != ABIType.host and target_soc:
            model_opencl_output_bin_path = get_opencl_binary_output_path(
                library_name, target_abi, target_soc, serial_num
            )
            model_opencl_parameter_path = get_opencl_parameter_output_path(
                library_name, target_abi, target_soc, serial_num
            )

        # generate input data
        sh_commands.gen_random_input(
            model_output_dir,
            subgraphs[0][YAMLKeyword.input_tensors],
            subgraphs[0][YAMLKeyword.input_shapes],
            subgraphs[0][YAMLKeyword.validation_inputs_data],
            input_ranges=subgraphs[0][YAMLKeyword.input_ranges],
            input_data_types=subgraphs[0][YAMLKeyword.input_data_types])

        runtime_list = []
        if target_abi == ABIType.host:
            runtime_list.extend([RuntimeType.cpu])
        elif model_runtime == RuntimeType.cpu_gpu:
            runtime_list.extend([RuntimeType.cpu, RuntimeType.gpu])
        else:
            runtime_list.extend([model_runtime])
        for runtime in runtime_list:
            device_type = parse_device_type(runtime)
            # run for specified soc
            run_output = sh_commands.tuning_run(
                abi=target_abi,
                serialno=serial_num,
                target_dir=build_tmp_binary_dir,
                target_name=target_name,
                vlog_level=flags.vlog_level,
                embed_model_data=embed_model_data,
                model_output_dir=model_output_dir,
                input_nodes=subgraphs[0][YAMLKeyword.input_tensors],
                output_nodes=subgraphs[0][YAMLKeyword.output_tensors],
                input_shapes=subgraphs[0][YAMLKeyword.input_shapes],
                output_shapes=subgraphs[0][YAMLKeyword.output_shapes],
                mace_model_dir=mace_model_dir,
                model_tag=model_name,
                device_type=device_type,
                running_round=flags.round,
                restart_round=flags.restart_round,
                limit_opencl_kernel_time=model_config[YAMLKeyword.limit_opencl_kernel_time],  # noqa
                tuning=False,
                out_of_range_check=flags.gpu_out_of_range_check,
                phone_data_dir=PHONE_DATA_DIR,
                model_graph_format=configs[YAMLKeyword.model_graph_format],
                omp_num_threads=flags.omp_num_threads,
                cpu_affinity_policy=flags.cpu_affinity_policy,
                gpu_perf_hint=flags.gpu_perf_hint,
                gpu_priority_hint=flags.gpu_priority_hint,
                runtime_failure_ratio=flags.runtime_failure_ratio,
                address_sanitizer=flags.address_sanitizer,
                opencl_binary_file=model_opencl_output_bin_path,
                opencl_parameter_file=model_opencl_parameter_path,
                libmace_dynamic_library_path=LIBMACE_DYNAMIC_PATH,
                link_dynamic=link_dynamic,
            )
            if flags.validate:
                model_file_path, weight_file_path = get_model_files(
                    model_config[YAMLKeyword.model_file_path],
                    model_config[YAMLKeyword.model_sha256_checksum],
                    BUILD_DOWNLOADS_DIR,
                    model_config[YAMLKeyword.weight_file_path],
                    model_config[YAMLKeyword.weight_sha256_checksum])

                sh_commands.validate_model(
                    abi=target_abi,
                    serialno=serial_num,
                    model_file_path=model_file_path,
                    weight_file_path=weight_file_path,
                    platform=model_config[YAMLKeyword.platform],
                    device_type=device_type,
                    input_nodes=subgraphs[0][YAMLKeyword.input_tensors],
                    output_nodes=subgraphs[0][YAMLKeyword.output_tensors],
                    input_shapes=subgraphs[0][YAMLKeyword.input_shapes],
                    output_shapes=subgraphs[0][YAMLKeyword.output_shapes],
                    model_output_dir=model_output_dir,
                    phone_data_dir=PHONE_DATA_DIR,
                    input_data_types=subgraphs[0][YAMLKeyword.input_data_types],  # noqa
                    caffe_env=flags.caffe_env,
                    validation_threshold=subgraphs[0][YAMLKeyword.validation_threshold][device_type])  # noqa
            if flags.report and flags.round > 0:
                tuned = is_tuned and device_type == DeviceType.GPU
                report_run_statistics(
                    run_output, target_abi, serial_num,
                    model_name, device_type, flags.report_dir,
                    tuned)

    if model_output_dirs:
        opencl_output_bin_path = get_opencl_binary_output_path(
            library_name, target_abi, target_soc, serial_num
        )
        opencl_parameter_bin_path = get_opencl_parameter_output_path(
            library_name, target_abi, target_soc, serial_num
        )
        # clear opencl output dir
        if os.path.exists(opencl_output_bin_path):
            sh.rm('-rf', opencl_output_bin_path)
        if os.path.exists(opencl_parameter_bin_path):
            sh.rm('-rf', opencl_parameter_bin_path)

        # merge all models' OpenCL binaries together
        sh_commands.merge_opencl_binaries(
            model_output_dirs, CL_COMPILED_BINARY_FILE_NAME,
            opencl_output_bin_path)
        # merge all models' OpenCL parameters together
        sh_commands.merge_opencl_parameters(
            model_output_dirs, CL_TUNED_PARAMETER_FILE_NAME,
            opencl_parameter_bin_path)


def run_quantize_stat(flags, configs):
    library_name = configs[YAMLKeyword.library_name]
    build_tmp_binary_dir = get_build_binary_dir(library_name, ABIType.host)

    for model_name in configs[YAMLKeyword.models]:
        check_model_converted(library_name, model_name,
                              configs[YAMLKeyword.model_graph_format],
                              configs[YAMLKeyword.model_data_format],
                              ABIType.host)
        MaceLogger.header(
            StringFormatter.block(
                "Run model %s on %s" % (model_name, ABIType.host)))

        model_config = configs[YAMLKeyword.models][model_name]
        subgraphs = model_config[YAMLKeyword.subgraphs]

        _, _, mace_model_dir = \
            get_build_model_dirs(library_name, model_name, ABIType.host,
                                 None, None,
                                 model_config[YAMLKeyword.model_file_path])

        mace_model_path = ""
        if configs[YAMLKeyword.model_graph_format] == ModelFormat.file:
            mace_model_path = "%s/%s.pb" % (mace_model_dir, model_name)

        p = subprocess.Popen(
            [
                "env",
                "MACE_CPP_MIN_VLOG_LEVEL=%s" % flags.vlog_level,
                "MACE_LOG_TENSOR_RANGE=1",
                "%s/%s" % (build_tmp_binary_dir, "quantize_stat"),
                "--model_name=%s" % model_name,
                "--input_node=%s" % ",".join(
                    subgraphs[0][YAMLKeyword.input_tensors]),
                "--output_node=%s" % ",".join(
                    subgraphs[0][YAMLKeyword.output_tensors]),
                "--input_shape=%s" % ":".join(
                    subgraphs[0][YAMLKeyword.input_shapes]),
                "--output_shape=%s" % ":".join(
                    subgraphs[0][YAMLKeyword.output_shapes]),
                "--input_dir=%s" % flags.input_dir,
                "--model_data_file=%s/%s.data" % (mace_model_dir, model_name),
                "--omp_num_threads=%s" % flags.omp_num_threads,
                "--model_file=%s" % mace_model_path,
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE)
        out, err = p.communicate()
        stdout = err + out
        print stdout
        print("Running finished!\n")


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

    if flags.quantize_stat:
        build_quantize_stat(configs)
        run_quantize_stat(flags, configs)
        return

    target_socs = configs[YAMLKeyword.target_socs]
    if not target_socs or ALL_SOC_TAG in target_socs:
        target_socs = sh_commands.adb_get_all_socs()

    for target_abi in configs[YAMLKeyword.target_abis]:
        # build target
        if flags.example:
            build_example(configs, target_abi,
                          not flags.disable_openmp,
                          flags.mace_lib_type)
        else:
            build_mace_run(configs, target_abi,
                           not flags.disable_openmp,
                           flags.address_sanitizer,
                           flags.mace_lib_type)

        # run
        if target_abi == ABIType.host:
            run_specific_target(flags, configs, target_abi, None, None)
        else:
            for target_soc in target_socs:
                serial_nums = \
                    sh_commands.get_target_socs_serialnos([target_soc])
                mace_check(serial_nums,
                           ModuleName.RUN,
                           'There is no device with soc: ' + target_soc)
                for serial_num in serial_nums:
                    with sh_commands.device_lock(serial_num):
                        run_specific_target(flags, configs, target_abi,
                                            target_soc, serial_num)

    # package the output files
    package_path = sh_commands.packaging_lib(BUILD_OUTPUT_DIR,
                                             configs[YAMLKeyword.library_name])
    print_package_summary(package_path)


################################
#  benchmark model
################################
def build_benchmark_model(configs, target_abi, enable_openmp, mace_lib_type):
    library_name = configs[YAMLKeyword.library_name]
    hexagon_mode = get_hexagon_mode(configs)

    link_dynamic = mace_lib_type == MACELibType.dynamic
    if link_dynamic:
        symbol_hidden = False
        benchmark_target = BM_MODEL_DYNAMIC_TARGET
    else:
        symbol_hidden = True
        benchmark_target = BM_MODEL_STATIC_TARGET

    build_arg = ""
    if configs[YAMLKeyword.model_graph_format] == ModelFormat.code:
        mace_check(os.path.exists(ENGINE_CODEGEN_DIR),
                   ModuleName.BENCHMARK,
                   "You should convert model first.")
        build_arg = "--per_file_copt=mace/benchmark/benchmark_model.cc@-DMODEL_GRAPH_FORMAT_CODE"  # noqa

    sh_commands.bazel_build(benchmark_target,
                            abi=target_abi,
                            enable_openmp=enable_openmp,
                            enable_opencl=get_opencl_mode(configs),
                            hexagon_mode=hexagon_mode,
                            symbol_hidden=symbol_hidden,
                            extra_args=build_arg)
    # clear tmp binary dir
    build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)
    if os.path.exists(build_tmp_binary_dir):
        sh.rm("-rf", build_tmp_binary_dir)
    os.makedirs(build_tmp_binary_dir)

    target_bin = "/".join(sh_commands.bazel_target_to_bin(benchmark_target))
    sh.cp("-f", target_bin, build_tmp_binary_dir)


def bm_specific_target(flags, configs, target_abi, target_soc, serial_num):
    library_name = configs[YAMLKeyword.library_name]
    embed_model_data = \
        configs[YAMLKeyword.model_data_format] == ModelFormat.code
    opencl_output_bin_path = ""
    opencl_parameter_path = ""
    link_dynamic = flags.mace_lib_type == MACELibType.dynamic

    if link_dynamic:
        bm_model_binary_name = BM_MODEL_DYNAMIC_NAME
    else:
        bm_model_binary_name = BM_MODEL_STATIC_NAME
    build_tmp_binary_dir = get_build_binary_dir(library_name, target_abi)

    if configs[YAMLKeyword.target_socs] and target_abi != ABIType.host:
        opencl_output_bin_path = get_opencl_binary_output_path(
            library_name, target_abi, target_soc, serial_num
        )
        opencl_parameter_path = get_opencl_parameter_output_path(
            library_name, target_abi, target_soc, serial_num
        )

    for model_name in configs[YAMLKeyword.models]:
        check_model_converted(library_name, model_name,
                              configs[YAMLKeyword.model_graph_format],
                              configs[YAMLKeyword.model_data_format],
                              target_abi)
        if target_abi == ABIType.host:
            device_name = ABIType.host
        else:
            device_name = \
                sh_commands.adb_get_device_name_by_serialno(serial_num)
        MaceLogger.header(
            StringFormatter.block(
                "Benchmark model %s on %s" % (model_name, device_name)))
        model_config = configs[YAMLKeyword.models][model_name]
        model_runtime = model_config[YAMLKeyword.runtime]
        subgraphs = model_config[YAMLKeyword.subgraphs]

        if not configs[YAMLKeyword.target_socs] or target_abi == ABIType.host:
            model_output_base_dir, model_output_dir, mace_model_dir = \
                get_build_model_dirs(library_name, model_name, target_abi,
                                     None, None,
                                     model_config[YAMLKeyword.model_file_path])
        else:
            model_output_base_dir, model_output_dir, mace_model_dir = \
                get_build_model_dirs(library_name, model_name, target_abi,
                                     target_soc, serial_num,
                                     model_config[YAMLKeyword.model_file_path])
        if os.path.exists(model_output_dir):
            sh.rm("-rf", model_output_dir)
        os.makedirs(model_output_dir)

        if target_abi != ABIType.host:
            sh_commands.clear_phone_data_dir(serial_num, PHONE_DATA_DIR)

        sh_commands.gen_random_input(
            model_output_dir,
            subgraphs[0][YAMLKeyword.input_tensors],
            subgraphs[0][YAMLKeyword.input_shapes],
            subgraphs[0][YAMLKeyword.validation_inputs_data],
            input_ranges=subgraphs[0][YAMLKeyword.input_ranges],
            input_data_types=subgraphs[0][YAMLKeyword.input_data_types])
        runtime_list = []
        if target_abi == ABIType.host:
            runtime_list.extend([RuntimeType.cpu])
        elif model_runtime == RuntimeType.cpu_gpu:
            runtime_list.extend([RuntimeType.cpu, RuntimeType.gpu])
        else:
            runtime_list.extend([model_runtime])
        for runtime in runtime_list:
            device_type = parse_device_type(runtime)
            sh_commands.benchmark_model(
                abi=target_abi,
                serialno=serial_num,
                benchmark_binary_dir=build_tmp_binary_dir,
                benchmark_binary_name=bm_model_binary_name,
                vlog_level=0,
                embed_model_data=embed_model_data,
                model_output_dir=model_output_dir,
                input_nodes=subgraphs[0][YAMLKeyword.input_tensors],
                output_nodes=subgraphs[0][YAMLKeyword.output_tensors],
                input_shapes=subgraphs[0][YAMLKeyword.input_shapes],
                output_shapes=subgraphs[0][YAMLKeyword.output_shapes],
                mace_model_dir=mace_model_dir,
                model_tag=model_name,
                device_type=device_type,
                phone_data_dir=PHONE_DATA_DIR,
                model_graph_format=configs[YAMLKeyword.model_graph_format],
                omp_num_threads=flags.omp_num_threads,
                cpu_affinity_policy=flags.cpu_affinity_policy,
                gpu_perf_hint=flags.gpu_perf_hint,
                gpu_priority_hint=flags.gpu_priority_hint,
                opencl_binary_file=opencl_output_bin_path,
                opencl_parameter_file=opencl_parameter_path,
                libmace_dynamic_library_path=LIBMACE_DYNAMIC_PATH,
                link_dynamic=link_dynamic)


def benchmark_model(flags):
    configs = format_model_config(flags)

    clear_build_dirs(configs[YAMLKeyword.library_name])

    target_socs = configs[YAMLKeyword.target_socs]
    if not target_socs or ALL_SOC_TAG in target_socs:
        target_socs = sh_commands.adb_get_all_socs()

    for target_abi in configs[YAMLKeyword.target_abis]:
        # build benchmark_model binary
        build_benchmark_model(configs, target_abi,
                              not flags.disable_openmp,
                              flags.mace_lib_type)

        if target_abi == ABIType.host:
            bm_specific_target(flags, configs, target_abi, None, None)
        else:
            for target_soc in target_socs:
                serial_nums = \
                    sh_commands.get_target_socs_serialnos([target_soc])
                mace_check(serial_nums,
                           ModuleName.BENCHMARK,
                           'There is no device with soc: ' + target_soc)
                for serial_num in serial_nums:
                    with sh_commands.device_lock(serial_num):
                        bm_specific_target(flags, configs, target_abi,
                                           target_soc, serial_num)


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
    convert_run_parent_parser = argparse.ArgumentParser(add_help=False)
    convert_run_parent_parser.add_argument(
        '--address_sanitizer',
        action="store_true",
        help="Whether to use address sanitizer to check memory error")
    run_bm_parent_parser = argparse.ArgumentParser(add_help=False)
    run_bm_parent_parser.add_argument(
        "--mace_lib_type",
        type=str_to_mace_lib_type,
        default=DefaultValues.mace_lib_type,
        help="[static | dynamic], Which type MACE library to use.")
    run_bm_parent_parser.add_argument(
        "--disable_openmp",
        action="store_true",
        help="Disable openmp for multiple thread.")
    run_bm_parent_parser.add_argument(
        "--omp_num_threads",
        type=int,
        default=DefaultValues.omp_num_threads,
        help="num of openmp threads")
    run_bm_parent_parser.add_argument(
        "--cpu_affinity_policy",
        type=int,
        default=DefaultValues.cpu_affinity_policy,
        help="0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY")
    run_bm_parent_parser.add_argument(
        "--gpu_perf_hint",
        type=int,
        default=DefaultValues.gpu_perf_hint,
        help="0:DEFAULT/1:LOW/2:NORMAL/3:HIGH")
    run_bm_parent_parser.add_argument(
        "--gpu_priority_hint",
        type=int,
        default=DefaultValues.gpu_priority_hint,
        help="0:DEFAULT/1:LOW/2:NORMAL/3:HIGH")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    convert = subparsers.add_parser(
        'convert',
        parents=[all_type_parent_parser, convert_run_parent_parser],
        help='convert to mace model (file or code)')
    convert.set_defaults(func=convert_func)
    run = subparsers.add_parser(
        'run',
        parents=[all_type_parent_parser, run_bm_parent_parser,
                 convert_run_parent_parser],
        help='run model in command line')
    run.set_defaults(func=run_mace)
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
        "--validate",
        action="store_true",
        help="whether to verify the results are consistent with "
             "the frameworks.")
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
        "--example",
        action="store_true",
        help="whether to run example.")
    run.add_argument(
        "--quantize_stat",
        action="store_true",
        help="whether to stat quantization range.")
    run.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="quantize stat input dir.")
    benchmark = subparsers.add_parser(
        'benchmark',
        parents=[all_type_parent_parser, run_bm_parent_parser],
        help='benchmark model for detail information')
    benchmark.set_defaults(func=benchmark_model)
    return parser.parse_known_args()


if __name__ == "__main__":
    flags, unparsed = parse_args()
    flags.func(flags)
