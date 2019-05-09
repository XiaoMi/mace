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

import enum
import hashlib
import inspect
import re
import os

import six


################################
# log
################################
class CMDColors:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_frame_info(level=2):
    caller_frame = inspect.stack()[level]
    info = inspect.getframeinfo(caller_frame[0])
    return info.filename + ':' + str(info.lineno) + ': '


class MaceLogger:
    @staticmethod
    def header(message):
        six.print_(CMDColors.PURPLE + message + CMDColors.ENDC)

    @staticmethod
    def summary(message):
        six.print_(CMDColors.GREEN + message + CMDColors.ENDC)

    @staticmethod
    def info(message):
        six.print_(get_frame_info() + message)

    @staticmethod
    def warning(message):
        six.print_(CMDColors.YELLOW + 'WARNING:' + get_frame_info() + message
                   + CMDColors.ENDC)

    @staticmethod
    def error(module, message, location_info=""):
        if not location_info:
            location_info = get_frame_info()
        six.print_(CMDColors.RED + 'ERROR: [' + module + '] ' + location_info
                   + message + CMDColors.ENDC)
        exit(1)


def mace_check(condition, module, message):
    if not condition:
        MaceLogger.error(module, message, get_frame_info())


################################
# String Formatter
################################
class StringFormatter:
    @staticmethod
    def table(header, data, title, align="R"):
        data_size = len(data)
        column_size = len(header)
        column_length = [len(str(ele)) + 1 for ele in header]
        for row_idx in range(data_size):
            data_tuple = data[row_idx]
            ele_size = len(data_tuple)
            assert (ele_size == column_size)
            for i in range(ele_size):
                column_length[i] = max(column_length[i],
                                       len(str(data_tuple[i])) + 1)

        table_column_length = sum(column_length) + column_size + 1
        dash_line = '-' * table_column_length + '\n'
        header_line = '=' * table_column_length + '\n'
        output = ""
        output += dash_line
        output += str(title).center(table_column_length) + '\n'
        output += dash_line
        output += '|' + '|'.join([str(header[i]).center(column_length[i])
                                  for i in range(column_size)]) + '|\n'
        output += header_line

        for data_tuple in data:
            ele_size = len(data_tuple)
            row_list = []
            for i in range(ele_size):
                if align == "R":
                    row_list.append(str(data_tuple[i]).rjust(column_length[i]))
                elif align == "L":
                    row_list.append(str(data_tuple[i]).ljust(column_length[i]))
                elif align == "C":
                    row_list.append(str(data_tuple[i])
                                    .center(column_length[i]))
            output += '|' + '|'.join(row_list) + "|\n" + dash_line
        return output

    @staticmethod
    def block(message):
        line_length = 10 + len(str(message)) + 10
        star_line = '*' * line_length + '\n'
        return star_line + str(message).center(line_length) + '\n' + star_line


################################
# definitions
################################
class DeviceType(object):
    CPU = 'CPU'
    GPU = 'GPU'
    HEXAGON = 'HEXAGON'
    HTA = 'HTA'
    APU = 'APU'

    # for validation threshold
    QUANTIZE = 'QUANTIZE'


class DataFormat(object):
    NONE = "NONE"
    NHWC = "NHWC"
    NCHW = "NCHW"
    OIHW = "OIHW"


################################
# Argument types
################################
class CaffeEnvType(enum.Enum):
    DOCKER = 0,
    LOCAL = 1,


################################
# common functions
################################
def formatted_file_name(input_file_name, input_name):
    res = input_file_name + '_'
    for c in input_name:
        res += c if c.isalnum() else '_'
    return res


def md5sum(s):
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()


def get_build_binary_dir(library_name, target_abi):
    return "%s/%s/%s/%s" % (
        BUILD_OUTPUT_DIR, library_name, BUILD_TMP_DIR_NAME, target_abi)


def get_model_lib_output_path(library_name, abi):
    lib_output_path = os.path.join(BUILD_OUTPUT_DIR, library_name,
                                   MODEL_OUTPUT_DIR_NAME, abi,
                                   "%s.a" % library_name)
    return lib_output_path


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


def parse_device_type(runtime):
    device_type = ""

    if runtime == RuntimeType.dsp:
        device_type = DeviceType.HEXAGON
    elif runtime == RuntimeType.hta:
        device_type = DeviceType.HTA
    elif runtime == RuntimeType.gpu:
        device_type = DeviceType.GPU
    elif runtime == RuntimeType.cpu:
        device_type = DeviceType.CPU
    elif runtime == RuntimeType.apu:
        device_type = DeviceType.APU

    return device_type


def sha256_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def get_dockerfile_info(dockerfile_path="",
                        dockerfile_sha256_checksum="",
                        docker_image_tag=""):
    dockerfile_local_path = ""
    if dockerfile_path.startswith("http://") or \
            dockerfile_path.startswith("https://"):
        dockerfile_local_path = \
            "third_party/caffe/" + docker_image_tag
        dockerfile = dockerfile_local_path + "/Dockerfile"
        if not os.path.exists(dockerfile_local_path):
            os.makedirs(dockerfile_local_path)
        if not os.path.exists(dockerfile) or \
                sha256_checksum(dockerfile) != dockerfile_sha256_checksum:
            MaceLogger.info("Downloading Dockerfile, please wait ...")
            six.moves.urllib.request.urlretrieve(dockerfile_path, dockerfile)
            MaceLogger.info("Dockerfile downloaded successfully.")

    if dockerfile_local_path:
        if sha256_checksum(dockerfile) != dockerfile_sha256_checksum:
            MaceLogger.error(ModuleName.MODEL_CONVERTER,
                             "Dockerfile sha256checksum not match")
    else:
        dockerfile_local_path = "third_party/caffe"
        docker_image_tag = "lastest"

    return dockerfile_local_path, docker_image_tag


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
            six.moves.urllib.request.urlretrieve(model_file_path, model_file)
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
            six.moves.urllib.request.urlretrieve(weight_file_path, weight_file)
            MaceLogger.info("Model weight downloaded successfully.")

    if weight_file:
        if sha256_checksum(weight_file) != weight_sha256_checksum:
            MaceLogger.error(ModuleName.MODEL_CONVERTER,
                             "weight file sha256checksum not match")

    return model_file, weight_file


def get_opencl_binary_output_path(library_name, target_abi, device):
    target_soc = device.target_socs
    device_name = device.device_name
    return '%s/%s/%s/%s/%s_%s.%s.%s.bin' % \
           (BUILD_OUTPUT_DIR,
            library_name,
            OUTPUT_OPENCL_BINARY_DIR_NAME,
            target_abi,
            library_name,
            OUTPUT_OPENCL_BINARY_FILE_NAME,
            device_name,
            target_soc)


def get_opencl_parameter_output_path(library_name, target_abi, device):
    target_soc = device.target_socs
    device_name = device.device_name
    return '%s/%s/%s/%s/%s_%s.%s.%s.bin' % \
           (BUILD_OUTPUT_DIR,
            library_name,
            OUTPUT_OPENCL_BINARY_DIR_NAME,
            target_abi,
            library_name,
            OUTPUT_OPENCL_PARAMETER_FILE_NAME,
            device_name,
            target_soc)


def get_build_model_dirs(library_name,
                         model_name,
                         target_abi,
                         device,
                         model_file_path):
    device_name = device.device_name
    target_socs = device.target_socs
    model_path_digest = md5sum(model_file_path)
    model_output_base_dir = '{}/{}/{}/{}/{}'.format(
        BUILD_OUTPUT_DIR, library_name, BUILD_TMP_DIR_NAME,
        model_name, model_path_digest)

    if target_abi == ABIType.host:
        model_output_dir = '%s/%s' % (model_output_base_dir, target_abi)
    elif not target_socs or not device.address:
        model_output_dir = '%s/%s/%s' % (model_output_base_dir,
                                         BUILD_TMP_GENERAL_OUTPUT_DIR_NAME,
                                         target_abi)
    else:
        model_output_dir = '{}/{}_{}/{}'.format(
            model_output_base_dir,
            device_name,
            target_socs,
            target_abi
        )

    mace_model_dir = '{}/{}/{}'.format(
        BUILD_OUTPUT_DIR, library_name, MODEL_OUTPUT_DIR_NAME
    )

    return model_output_base_dir, model_output_dir, mace_model_dir


def abi_to_internal(abi):
    if abi in [ABIType.armeabi_v7a, ABIType.arm64_v8a]:
        return abi
    if abi == ABIType.arm64:
        return ABIType.aarch64
    if abi == ABIType.armhf:
        return ABIType.armeabi_v7a


def infer_toolchain(abi):
    if abi in [ABIType.armeabi_v7a, ABIType.arm64_v8a]:
        return ToolchainType.android
    if abi == ABIType.armhf:
        return ToolchainType.arm_linux_gnueabihf
    if abi == ABIType.arm64:
        return ToolchainType.aarch64_linux_gnu
    return ''


################################
# YAML key word
################################
class YAMLKeyword(object):
    library_name = 'library_name'
    target_abis = 'target_abis'
    target_socs = 'target_socs'
    model_graph_format = 'model_graph_format'
    model_data_format = 'model_data_format'
    models = 'models'
    platform = 'platform'
    device_name = 'device_name'
    system = 'system'
    address = 'address'
    username = 'username'
    password = 'password'
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
    check_tensors = 'check_tensors'
    check_shapes = 'check_shapes'
    runtime = 'runtime'
    data_type = 'data_type'
    input_data_types = 'input_data_types'
    output_data_types = 'output_data_types'
    input_data_formats = 'input_data_formats'
    output_data_formats = 'output_data_formats'
    limit_opencl_kernel_time = 'limit_opencl_kernel_time'
    nnlib_graph_mode = 'nnlib_graph_mode'
    obfuscate = 'obfuscate'
    winograd = 'winograd'
    quantize = 'quantize'
    quantize_large_weights = 'quantize_large_weights'
    quantize_range_file = 'quantize_range_file'
    change_concat_ranges = 'change_concat_ranges'
    validation_inputs_data = 'validation_inputs_data'
    validation_threshold = 'validation_threshold'
    graph_optimize_options = 'graph_optimize_options'  # internal use for now
    cl_mem_type = 'cl_mem_type'
    backend = 'backend'
    validation_outputs_data = 'validation_outputs_data'
    accuracy_validation_script = 'accuracy_validation_script'
    docker_image_tag = 'docker_image_tag'
    dockerfile_path = 'dockerfile_path'
    dockerfile_sha256_checksum = 'dockerfile_sha256_checksum'


################################
# SystemType
################################
class SystemType:
    host = 'host'
    android = 'android'
    arm_linux = 'arm_linux'


################################
# common device str
################################

PHONE_DATA_DIR = '/data/local/tmp/mace_run'
DEVICE_DATA_DIR = '/tmp/data/mace_run'
DEVICE_INTERIOR_DIR = PHONE_DATA_DIR + "/interior"
BUILD_OUTPUT_DIR = 'build'
BUILD_TMP_DIR_NAME = '_tmp'
BUILD_DOWNLOADS_DIR = BUILD_OUTPUT_DIR + '/downloads'
BUILD_TMP_GENERAL_OUTPUT_DIR_NAME = 'general'
MODEL_OUTPUT_DIR_NAME = 'model'
EXAMPLE_STATIC_NAME = "example_static"
EXAMPLE_DYNAMIC_NAME = "example_dynamic"
EXAMPLE_STATIC_TARGET = "//examples/cli:" + EXAMPLE_STATIC_NAME
EXAMPLE_DYNAMIC_TARGET = "//examples/cli:" + EXAMPLE_DYNAMIC_NAME
MACE_RUN_STATIC_NAME = "mace_run_static"
MACE_RUN_DYNAMIC_NAME = "mace_run_dynamic"
MACE_RUN_STATIC_TARGET = "//mace/tools/validation:" + MACE_RUN_STATIC_NAME
MACE_RUN_DYNAMIC_TARGET = "//mace/tools/validation:" + MACE_RUN_DYNAMIC_NAME
CL_COMPILED_BINARY_FILE_NAME = "mace_cl_compiled_program.bin"
BUILD_TMP_OPENCL_BIN_DIR = 'opencl_bin'
LIBMACE_DYNAMIC_PATH = "bazel-bin/mace/libmace/libmace.so"
CL_TUNED_PARAMETER_FILE_NAME = "mace_run.config"
MODEL_HEADER_DIR_PATH = 'include/mace/public'
OUTPUT_LIBRARY_DIR_NAME = 'lib'
OUTPUT_OPENCL_BINARY_DIR_NAME = 'opencl'
OUTPUT_OPENCL_BINARY_FILE_NAME = 'compiled_opencl_kernel'
OUTPUT_OPENCL_PARAMETER_FILE_NAME = 'tuned_opencl_parameter'
CODEGEN_BASE_DIR = 'mace/codegen'
MODEL_CODEGEN_DIR = CODEGEN_BASE_DIR + '/models'
ENGINE_CODEGEN_DIR = CODEGEN_BASE_DIR + '/engine'
LIB_CODEGEN_DIR = CODEGEN_BASE_DIR + '/lib'
OPENCL_CODEGEN_DIR = CODEGEN_BASE_DIR + '/opencl'
LIBMACE_SO_TARGET = "//mace/libmace:libmace.so"
LIBMACE_STATIC_TARGET = "//mace/libmace:libmace_static"
LIBMACE_STATIC_PATH = "bazel-genfiles/mace/libmace/libmace.a"
MODEL_LIB_TARGET = "//mace/codegen:generated_models"
MODEL_LIB_PATH = "bazel-bin/mace/codegen/libgenerated_models.a"
QUANTIZE_STAT_TARGET = "//mace/tools/quantization:quantize_stat"
BM_MODEL_STATIC_NAME = "benchmark_model_static"
BM_MODEL_DYNAMIC_NAME = "benchmark_model_dynamic"
BM_MODEL_STATIC_TARGET = "//mace/tools/benchmark:" + BM_MODEL_STATIC_NAME
BM_MODEL_DYNAMIC_TARGET = "//mace/tools/benchmark:" + BM_MODEL_DYNAMIC_NAME


################################
# Model File Format
################################
class ModelFormat(object):
    file = 'file'
    code = 'code'


################################
# ABI Type
################################
class ABIType(object):
    armeabi_v7a = 'armeabi-v7a'
    arm64_v8a = 'arm64-v8a'
    arm64 = 'arm64'
    aarch64 = 'aarch64'
    armhf = 'armhf'
    host = 'host'


################################
# Module name
################################
class ModuleName(object):
    YAML_CONFIG = 'YAML CONFIG'
    MODEL_CONVERTER = 'Model Converter'
    RUN = 'RUN'
    BENCHMARK = 'Benchmark'


#################################
# mace lib type
#################################
class MACELibType(object):
    static = 0
    dynamic = 1


#################################
# Run time type
#################################
class RuntimeType(object):
    cpu = 'cpu'
    gpu = 'gpu'
    dsp = 'dsp'
    hta = 'hta'
    apu = 'apu'
    cpu_gpu = 'cpu+gpu'


#################################
# Tool chain Type
#################################
class ToolchainType:
    android = 'android'
    arm_linux_gnueabihf = 'arm_linux_gnueabihf'
    aarch64_linux_gnu = 'aarch64_linux_gnu'


#################################
# SOC tag
#################################
class TargetSOCTag:
    all = 'all'
    random = 'random'


def split_shape(shape):
    if shape.strip() == "":
        return []
    else:
        return shape.split(',')
