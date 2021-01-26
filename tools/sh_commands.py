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

import glob
import logging
import numpy as np
import os
import random
import re
import sh
import struct
import sys
import time
import platform

import six

import common
from common import abi_to_internal

sys.path.insert(0, "mace/python/tools")
try:
    from encrypt_opencl_codegen import encrypt_opencl_codegen
    from opencl_binary_codegen import generate_opencl_code
    from generate_data import generate_input_data
    from validate import validate
    from mace_engine_factory_codegen import gen_mace_engine_factory
except Exception as e:
    six.print_("Import error:\n%s" % e, file=sys.stderr)
    exit(1)


################################
# common
################################


def strip_invalid_utf8(str):
    return sh.iconv(str, "-c", "-t", "UTF-8")


def split_stdout(stdout_str):
    stdout_str = strip_invalid_utf8(stdout_str)
    # Filter out last empty line
    return [line.strip() for line in stdout_str.split('\n') if
            len(line.strip()) > 0]


def make_output_processor(buff):
    def process_output(line):
        six.print_(line.rstrip())
        buff.append(line)

    return process_output


def device_lock_path(serialno):
    return "/tmp/device-lock-%s" % serialno


def device_lock(serialno, timeout=7200):
    import filelock
    return filelock.FileLock(device_lock_path(serialno.replace("/", "")),
                             timeout=timeout)


def is_device_locked(serialno):
    import filelock
    try:
        with device_lock(serialno, timeout=0.000001):
            return False
    except filelock.Timeout:
        return True


class BuildType(object):
    proto = 'proto'
    code = 'code'


def stdout_success(stdout):
    stdout_lines = stdout.split("\n")
    for line in stdout_lines:
        if "Aborted" in line or "FAILED" in line or \
                "Segmentation fault" in line:
            return False
    return True


# select a random unlocked device support the ABI
def choose_a_random_device(target_devices, target_abi):
    eligible_devices = [dev for dev in target_devices
                        if target_abi in dev[common.YAMLKeyword.target_abis]]
    unlocked_devices = [dev for dev in eligible_devices if
                        not is_device_locked(dev[common.YAMLKeyword.address])]
    if len(unlocked_devices) > 0:
        chosen_devices = [random.choice(unlocked_devices)]
    else:
        chosen_devices = [random.choice(eligible_devices)]
    return chosen_devices


################################
# clear data
################################
def clear_phone_data_dir(serialno, phone_data_dir):
    sh.adb("-s",
           serialno,
           "shell",
           "rm -rf %s" % phone_data_dir)


################################
# adb commands
################################
def adb_devices():
    serialnos = []
    p = re.compile(r'(\S+)\s+device')
    for line in split_stdout(sh.adb("devices")):
        m = p.match(line)
        if m:
            serialnos.append(m.group(1))

    return serialnos


def get_soc_serialnos_map():
    serialnos = adb_devices()
    soc_serialnos_map = {}
    for serialno in serialnos:
        props = adb_getprop_by_serialno(serialno)
        soc_serialnos_map.setdefault(props["ro.board.platform"], []) \
            .append(serialno)

    return soc_serialnos_map


def get_target_socs_serialnos(target_socs=None):
    soc_serialnos_map = get_soc_serialnos_map()
    serialnos = []
    if target_socs is None:
        target_socs = soc_serialnos_map.keys()
    for target_soc in target_socs:
        serialnos.extend(soc_serialnos_map[target_soc])
    return serialnos


def adb_getprop_by_serialno(serialno):
    outputs = sh.adb("-s", serialno, "shell", "getprop")
    raw_props = split_stdout(outputs)
    props = {}
    p = re.compile(r'\[(.+)\]: \[(.+)\]')
    for raw_prop in raw_props:
        m = p.match(raw_prop)
        if m:
            props[m.group(1)] = m.group(2)
    return props


def adb_get_device_name_by_serialno(serialno):
    props = adb_getprop_by_serialno(serialno)
    return props.get("ro.product.model", "").replace(' ', '')


def adb_supported_abis(serialno):
    props = adb_getprop_by_serialno(serialno)
    abilist_str = props["ro.product.cpu.abilist"]
    abis = [abi.strip() for abi in abilist_str.split(',')]
    return abis


def adb_get_all_socs():
    socs = []
    for d in adb_devices():
        props = adb_getprop_by_serialno(d)
        socs.append(props["ro.board.platform"])
    return set(socs)


def adb_push(src_path, dst_path, serialno):
    sh.adb("-s", serialno, "push", src_path, dst_path)


def adb_pull(src_path, dst_path, serialno):
    try:
        sh.adb("-s", serialno, "pull", src_path, dst_path)
    except Exception as e:
        six.print_("Error msg: %s" % e, file=sys.stderr)


################################
# Toolchain
################################
def asan_rt_library_names(abi):
    asan_rt_names = {
        "armeabi-v7a": "libclang_rt.asan-arm-android.so",
        "arm64-v8a": "libclang_rt.asan-aarch64-android.so",
    }
    return asan_rt_names[abi]


def find_asan_rt_library(abi, asan_rt_path=''):
    if not asan_rt_path:
        find_path = os.environ['ANDROID_NDK_HOME']
        candidates = split_stdout(sh.find(find_path, "-name",
                                          asan_rt_library_names(abi)))
        if len(candidates) == 0:
            common.MaceLogger.error(
                "Toolchain",
                "Can't find AddressSanitizer runtime library in %s" %
                find_path)
        elif len(candidates) > 1:
            common.MaceLogger.info(
                "More than one AddressSanitizer runtime library, use the 1st")
        return candidates[0]
    return "%s/%s" % (asan_rt_path, asan_rt_library_names(abi))


def simpleperf_abi_dir_names(abi):
    simpleperf_dir_names = {
        "armeabi-v7a": "arm",
        "arm64-v8a": "arm64",
    }
    return simpleperf_dir_names[abi]


def find_simpleperf_library(abi, simpleperf_path=''):
    if not simpleperf_path:
        find_path = os.environ['ANDROID_NDK_HOME']
        candidates = split_stdout(sh.find(find_path, "-name", "simpleperf"))
        if len(candidates) == 0:
            common.MaceLogger.error(
                "Toolchain",
                "Can't find Simpleperf runtime library in % s" %
                find_path)
        found = False
        for candidate in candidates:
            if candidate.find(simpleperf_abi_dir_names(abi) + "/") != -1:
                found = True
                return candidate
        if not found:
            common.MaceLogger.error(
                "Toolchain",
                "Can't find Simpleperf runtime library in % s" %
                find_path)

    return "%s/simpleperf" % simpleperf_path


def get_apu_ancient(enable_apu):
    if (not enable_apu):
        return False
    common.mace_check(abi == "arm64-v8a", "",
                      "Only support arm64-v8a for apu runtime")
    target_props = sh_commands.adb_getprop_by_serialno(self.address)
    target_soc = target_props["ro.board.platform"]
    android_ver = (int)(target_props["ro.build.version.release"])


################################
# bazel commands
################################
def bazel_build(target,
                abi="armeabi-v7a",
                toolchain='android',
                enable_hexagon=False,
                enable_hta=False,
                enable_apu=False,
                apu_ancient=False,
                enable_neon=True,
                enable_opencl=True,
                enable_quantize=True,
                enable_bfloat16=False,
                enable_fp16=False,
                enable_rpcmem=False,
                address_sanitizer=False,
                symbol_hidden=True,
                debug_mode=False,
                extra_args=""):
    six.print_("* Build %s with ABI %s" % (target, abi))
    if abi == "host":
        toolchain = platform.system().lower()
        bazel_args = (
            "build",
            "--config",
            toolchain,
            "--define",
            "quantize=%s" % str(enable_quantize).lower(),
            "--define",
            "bfloat16=%s" % str(enable_bfloat16).lower(),
            target,
        )
    else:
        bazel_args = (
            "build",
            target,
            "--config",
            toolchain,
            "--cpu=%s" % abi_to_internal(abi),
            "--define",
            "neon=%s" % str(enable_neon).lower(),
            "--define",
            "opencl=%s" % str(enable_opencl).lower(),
            "--define",
            "quantize=%s" % str(enable_quantize).lower(),
            "--define",
            "bfloat16=%s" % str(enable_bfloat16).lower(),
            "--define",
            "fp16=%s" % str(enable_fp16).lower(),
            "--define",
            "rpcmem=%s" % str(enable_rpcmem).lower(),
            "--define",
            "hexagon=%s" % str(enable_hexagon).lower(),
            "--define",
            "hta=%s" % str(enable_hta).lower(),
            "--define",
            "apu=%s" % str(enable_apu).lower(),
            "--define",
            "apu_ancient=%s" % str(apu_ancient).lower())
    if address_sanitizer:
        bazel_args += ("--config", "asan")
    if debug_mode:
        bazel_args += ("--config", "debug")
    if not address_sanitizer and not debug_mode:
        if toolchain == "darwin" or toolchain == "ios":
            bazel_args += ("--config", "optimization_darwin")
        else:
            bazel_args += ("--config", "optimization")
        if symbol_hidden:
            bazel_args += ("--config", "symbol_hidden")
    if extra_args:
        bazel_args += (extra_args,)
        six.print_(bazel_args)
    sh.bazel(
        _fg=True,
        *bazel_args)
    six.print_(bazel_args)
    six.print_("Build done!\n")


def bazel_build_common(target, build_args=""):
    stdout_buff = []
    process_output = make_output_processor(stdout_buff)
    sh.bazel(
        "build",
        target + build_args,
        _tty_in=True,
        _out=process_output,
        _err_to_out=True)
    return "".join(stdout_buff)


def bazel_target_to_bin(target):
    # change //mace/a/b:c to bazel-bin/mace/a/b/c
    prefix, bin_name = target.split(':')
    prefix = prefix.replace('//', '/')
    if prefix.startswith('/'):
        prefix = prefix[1:]
    host_bin_path = "bazel-bin/%s" % prefix
    return host_bin_path, bin_name


################################
# mace commands
################################
def gen_encrypted_opencl_source(codegen_path="mace/codegen"):
    sh.mkdir("-p", "%s/opencl" % codegen_path)
    encrypt_opencl_codegen("./mace/ops/opencl/cl/",
                           "mace/codegen/opencl/opencl_encrypt_program.cc")


def gen_mace_engine_factory_source(model_tags,
                                   embed_model_data,
                                   codegen_path="mace/codegen"):
    six.print_("* Generate mace engine creator source")
    codegen_tools_dir = "%s/engine" % codegen_path
    sh.rm("-rf", codegen_tools_dir)
    sh.mkdir("-p", codegen_tools_dir)
    gen_mace_engine_factory(
        model_tags,
        embed_model_data,
        codegen_tools_dir)
    six.print_("Generate mace engine creator source done!\n")


def merge_opencl_binaries(binaries_dirs,
                          cl_compiled_program_file_name,
                          output_file_path):
    platform_info_key = 'mace_opencl_precompiled_platform_info_key'
    cl_bin_dirs = []
    for d in binaries_dirs:
        cl_bin_dirs.append(os.path.join(d, "opencl_bin"))
    # create opencl binary output dir
    opencl_binary_dir = os.path.dirname(output_file_path)
    if not os.path.exists(opencl_binary_dir):
        sh.mkdir("-p", opencl_binary_dir)
    kvs = {}
    for binary_dir in cl_bin_dirs:
        binary_path = os.path.join(binary_dir, cl_compiled_program_file_name)
        if not os.path.exists(binary_path):
            continue

        six.print_('generate opencl code from', binary_path)
        with open(binary_path, "rb") as f:
            binary_array = np.fromfile(f, dtype=np.uint8)

        idx = 0
        size, = struct.unpack("Q", binary_array[idx:idx + 8])
        idx += 8
        for _ in six.moves.range(size):
            key_size, = struct.unpack("i", binary_array[idx:idx + 4])
            idx += 4
            key, = struct.unpack(
                str(key_size) + "s", binary_array[idx:idx + key_size])
            idx += key_size
            value_size, = struct.unpack("i", binary_array[idx:idx + 4])
            idx += 4
            if key == platform_info_key and key in kvs:
                common.mace_check(
                    (kvs[key] == binary_array[idx:idx + value_size]).all(),
                    "",
                    "There exists more than one OpenCL version for models:"
                    " %s vs %s " %
                    (kvs[key], binary_array[idx:idx + value_size]))
            else:
                kvs[key] = binary_array[idx:idx + value_size]
            idx += value_size

    output_byte_array = bytearray()
    data_size = len(kvs)
    output_byte_array.extend(struct.pack("Q", data_size))
    for key, value in six.iteritems(kvs):
        key_size = len(key)
        output_byte_array.extend(struct.pack("i", key_size))
        output_byte_array.extend(struct.pack(str(key_size) + "s", key))
        value_size = len(value)
        output_byte_array.extend(struct.pack("i", value_size))
        output_byte_array.extend(value)

    np.array(output_byte_array).tofile(output_file_path)


def merge_opencl_parameters(binaries_dirs,
                            cl_parameter_file_name,
                            output_file_path):
    cl_bin_dirs = []
    for d in binaries_dirs:
        cl_bin_dirs.append(os.path.join(d, "opencl_bin"))
    # create opencl binary output dir
    opencl_binary_dir = os.path.dirname(output_file_path)
    if not os.path.exists(opencl_binary_dir):
        sh.mkdir("-p", opencl_binary_dir)
    kvs = {}
    for binary_dir in cl_bin_dirs:
        binary_path = os.path.join(binary_dir, cl_parameter_file_name)
        if not os.path.exists(binary_path):
            continue

        six.print_('generate opencl parameter from', binary_path)
        with open(binary_path, "rb") as f:
            binary_array = np.fromfile(f, dtype=np.uint8)

        idx = 0
        size, = struct.unpack("Q", binary_array[idx:idx + 8])
        idx += 8
        for _ in six.moves.range(size):
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
    for key, value in six.iteritems(kvs):
        key_size = len(key)
        output_byte_array.extend(struct.pack("i", key_size))
        output_byte_array.extend(struct.pack(str(key_size) + "s", key))
        value_size = len(value)
        output_byte_array.extend(struct.pack("i", value_size))
        output_byte_array.extend(value)

    np.array(output_byte_array).tofile(output_file_path)


def gen_input(model_output_dir,
              input_nodes,
              input_shapes,
              input_files=None,
              input_ranges=None,
              input_data_types=None,
              input_data_map=None,
              input_file_name="model_input"):
    for input_name in input_nodes:
        formatted_name = common.formatted_file_name(
            input_file_name, input_name)
        if os.path.exists("%s/%s" % (model_output_dir, formatted_name)):
            sh.rm("%s/%s" % (model_output_dir, formatted_name))
    input_file_list = []
    if isinstance(input_files, list):
        input_file_list.extend(input_files)
    else:
        input_file_list.append(input_files)
    if input_data_map:
        for i in range(len(input_nodes)):
            dst_input_file = model_output_dir + '/' + \
                             common.formatted_file_name(input_file_name,
                                                        input_nodes[i])
            input_name = input_nodes[i]
            common.mace_check(input_name in input_data_map,
                              common.ModuleName.RUN,
                              "The preprocessor API in PrecisionValidator"
                              " script should return all inputs of model")
            if input_data_types[i] == 'float32':
                input_data = np.array(input_data_map[input_name],
                                      dtype=np.float32)
            elif input_data_types[i] == 'int32':
                input_data = np.array(input_data_map[input_name],
                                      dtype=np.int32)
            else:
                common.mace_check(
                    False,
                    common.ModuleName.RUN,
                    'Do not support input data type %s' % input_data_types[i])
            common.mace_check(
                list(map(int, common.split_shape(input_shapes[i])))
                == list(input_data.shape),
                common.ModuleName.RUN,
                "The shape return from preprocessor API of"
                " PrecisionValidator script is not same with"
                " model deployment file. %s vs %s"
                % (str(input_shapes[i]), str(input_data.shape)))
            input_data.tofile(dst_input_file)
    elif len(input_file_list) != 0:
        input_name_list = []
        if isinstance(input_nodes, list):
            input_name_list.extend(input_nodes)
        else:
            input_name_list.append(input_nodes)
        common.mace_check(len(input_file_list) == len(input_name_list),
                          common.ModuleName.RUN,
                          'If input_files set, the input files should '
                          'match the input names.')
        for i in range(len(input_file_list)):
            if input_file_list[i] is not None:
                dst_input_file = model_output_dir + '/' + \
                                 common.formatted_file_name(input_file_name,
                                                            input_name_list[i])
                if input_file_list[i].startswith("http://") or \
                        input_file_list[i].startswith("https://"):
                    six.moves.urllib.request.urlretrieve(input_file_list[i],
                                                         dst_input_file)
                else:
                    sh.cp("-f", input_file_list[i], dst_input_file)
    else:
        # generate random input files
        input_nodes_str = ",".join(input_nodes)
        input_shapes_str = ":".join(input_shapes)
        input_ranges_str = ":".join(input_ranges)
        input_data_types_str = ",".join(input_data_types)
        generate_input_data("%s/%s" % (model_output_dir, input_file_name),
                            input_nodes_str,
                            input_shapes_str,
                            input_ranges_str,
                            input_data_types_str)


def gen_opencl_binary_cpps(opencl_bin_file_path,
                           opencl_param_file_path,
                           opencl_bin_cpp_path,
                           opencl_param_cpp_path):
    output_dir = os.path.dirname(opencl_bin_cpp_path)
    if not os.path.exists(output_dir):
        sh.mkdir("-p", output_dir)
    opencl_bin_load_func_name = 'LoadOpenCLBinary'
    opencl_bin_size_func_name = 'OpenCLBinarySize'
    opencl_param_load_func_name = 'LoadOpenCLParameter'
    opencl_param_size_func_name = 'OpenCLParameterSize'
    generate_opencl_code(opencl_bin_file_path, opencl_bin_load_func_name,
                         opencl_bin_size_func_name, opencl_bin_cpp_path)
    generate_opencl_code(opencl_param_file_path, opencl_param_load_func_name,
                         opencl_param_size_func_name, opencl_param_cpp_path)


def update_mace_run_binary(build_tmp_binary_dir, link_dynamic=False):
    if link_dynamic:
        mace_run_filepath = build_tmp_binary_dir + "/mace_run_dynamic"
    else:
        mace_run_filepath = build_tmp_binary_dir + "/mace_run_static"

    if os.path.exists(mace_run_filepath):
        sh.rm("-rf", mace_run_filepath)
    if link_dynamic:
        sh.cp("-f", "bazel-bin/mace/tools/mace_run_dynamic",
              build_tmp_binary_dir)
    else:
        sh.cp("-f", "bazel-bin/mace/tools/mace_run_static",
              build_tmp_binary_dir)


def create_internal_storage_dir(serialno, phone_data_dir):
    internal_storage_dir = "%s/interior/" % phone_data_dir
    sh.adb("-s", serialno, "shell", "mkdir", "-p", internal_storage_dir)
    return internal_storage_dir


def push_depended_so_libs(libmace_dynamic_library_path,
                          abi, phone_data_dir, serialno):
    src_file = "%s/sources/cxx-stl/llvm-libc++/libs/" \
               "%s/libc++_shared.so" \
               % (os.environ["ANDROID_NDK_HOME"], abi)
    try:
        dep_so_libs = sh.bash(os.environ["ANDROID_NDK_HOME"] + "/ndk-depends",
                              libmace_dynamic_library_path)
    except sh.ErrorReturnCode_127:
        print("Find no ndk-depends, use default libc++_shared.so")
    else:
        for dep in split_stdout(dep_so_libs):
            if dep == "libgnustl_shared.so":
                src_file = "%s/sources/cxx-stl/gnu-libstdc++/4.9/libs/" \
                           "%s/libgnustl_shared.so" \
                           % (os.environ["ANDROID_NDK_HOME"], abi)
    print("push %s to %s" % (src_file, phone_data_dir))
    adb_push(src_file, phone_data_dir, serialno)


def validate_model(abi,
                   device,
                   model_file_path,
                   weight_file_path,
                   docker_image_tag,
                   dockerfile_path,
                   platform,
                   device_type,
                   input_nodes,
                   output_nodes,
                   input_shapes,
                   output_shapes,
                   input_data_formats,
                   output_data_formats,
                   model_output_dir,
                   input_data_types,
                   caffe_env,
                   input_file_name="model_input",
                   output_file_name="model_out",
                   validation_threshold=0.9,
                   backend="tensorflow",
                   validation_outputs_data=[],
                   log_file=""):
    if not validation_outputs_data:
        six.print_("* Validate with %s" % platform)
    else:
        six.print_("* Validate with file: %s" % validation_outputs_data)
    if abi != "host":
        for output_name in output_nodes:
            formatted_name = common.formatted_file_name(
                output_file_name, output_name)
            if os.path.exists("%s/%s" % (model_output_dir,
                                         formatted_name)):
                sh.rm("-rf", "%s/%s" % (model_output_dir, formatted_name))
            device.pull_from_data_dir(formatted_name, model_output_dir)

    if platform in ["tensorflow", "onnx", "pytorch", 'keras']:
        validate(platform, model_file_path, "",
                 "%s/%s" % (model_output_dir, input_file_name),
                 "%s/%s" % (model_output_dir, output_file_name), device_type,
                 ":".join(input_shapes), ":".join(output_shapes),
                 ",".join(input_data_formats), ",".join(output_data_formats),
                 ",".join(input_nodes), ",".join(output_nodes),
                 validation_threshold, ",".join(input_data_types), backend,
                 validation_outputs_data,
                 log_file)
    elif platform == "caffe":
        image_name = "mace-caffe:" + docker_image_tag
        container_name = "mace_caffe_" + docker_image_tag + "_validator"

        if caffe_env == common.CaffeEnvType.LOCAL:
            try:
                import caffe
            except ImportError:
                logging.error('There is no caffe python module.')
            validate(platform, model_file_path, weight_file_path,
                     "%s/%s" % (model_output_dir, input_file_name),
                     "%s/%s" % (model_output_dir, output_file_name),
                     device_type,
                     ":".join(input_shapes), ":".join(output_shapes),
                     ",".join(input_data_formats),
                     ",".join(output_data_formats),
                     ",".join(input_nodes), ",".join(output_nodes),
                     validation_threshold, ",".join(input_data_types), backend,
                     validation_outputs_data,
                     log_file)
        elif caffe_env == common.CaffeEnvType.DOCKER:
            docker_image_id = sh.docker("images", "-q", image_name)
            if not docker_image_id:
                six.print_("Build caffe docker")
                sh.docker("build", "-t", image_name,
                          dockerfile_path)

            container_id = sh.docker("ps", "-qa", "-f",
                                     "name=%s" % container_name)
            if container_id and not sh.docker("ps", "-qa", "--filter",
                                              "status=running", "-f",
                                              "name=%s" % container_name):
                sh.docker("rm", "-f", container_name)
                container_id = ""
            if not container_id:
                six.print_("Run caffe container")
                sh.docker(
                    "run",
                    "-d",
                    "-it",
                    "--name",
                    container_name,
                    image_name,
                    "/bin/bash")

            for input_name in input_nodes:
                formatted_input_name = common.formatted_file_name(
                    input_file_name, input_name)
                sh.docker(
                    "cp",
                    "%s/%s" % (model_output_dir, formatted_input_name),
                    "%s:/mace" % container_name)

            for output_name in output_nodes:
                formatted_output_name = common.formatted_file_name(
                    output_file_name, output_name)
                sh.docker(
                    "cp",
                    "%s/%s" % (model_output_dir, formatted_output_name),
                    "%s:/mace" % container_name)
            model_file_name = os.path.basename(model_file_path)
            weight_file_name = os.path.basename(weight_file_path)
            sh.docker("cp", "tools/common.py", "%s:/mace" % container_name)
            sh.docker("cp", "tools/validate.py", "%s:/mace" % container_name)
            sh.docker("cp", model_file_path, "%s:/mace" % container_name)
            sh.docker("cp", weight_file_path, "%s:/mace" % container_name)

            sh.docker(
                "exec",
                container_name,
                "python",
                "-u",
                "/mace/validate.py",
                "--platform=caffe",
                "--model_file=/mace/%s" % model_file_name,
                "--weight_file=/mace/%s" % weight_file_name,
                "--input_file=/mace/%s" % input_file_name,
                "--mace_out_file=/mace/%s" % output_file_name,
                "--device_type=%s" % device_type,
                "--input_node=%s" % ",".join(input_nodes),
                "--output_node=%s" % ",".join(output_nodes),
                "--input_shape=%s" % ":".join(input_shapes),
                "--output_shape=%s" % ":".join(output_shapes),
                "--input_data_format=%s" % ",".join(input_data_formats),
                "--output_data_format=%s" % ",".join(output_data_formats),
                "--validation_threshold=%f" % validation_threshold,
                "--input_data_type=%s" % ",".join(input_data_types),
                "--backend=%s" % backend,
                "--validation_outputs_data=%s" % ",".join(
                    validation_outputs_data),
                "--log_file=%s" % log_file,
                _fg=True)
    elif platform == "megengine":
        validate(platform, model_file_path, "",
                 "%s/%s" % (model_output_dir, input_file_name),
                 "%s/%s" % (model_output_dir, output_file_name),
                 device_type,
                 ":".join(input_shapes), ":".join(output_shapes),
                 ",".join(input_data_formats),
                 ",".join(output_data_formats),
                 ",".join(input_nodes), ",".join(output_nodes),
                 validation_threshold, ",".join(input_data_types), backend,
                 validation_outputs_data,
                 log_file)

    six.print_("Validation done!\n")


################################
# library
################################
def packaging_lib(libmace_output_dir, project_name):
    six.print_("* Package libs for %s" % project_name)
    tar_package_name = "libmace_%s.tar.gz" % project_name
    project_dir = "%s/%s" % (libmace_output_dir, project_name)
    tar_package_path = "%s/%s" % (project_dir, tar_package_name)
    if os.path.exists(tar_package_path):
        sh.rm("-rf", tar_package_path)

    six.print_("Start packaging '%s' libs into %s" % (project_name,
                                                      tar_package_path))
    which_sys = platform.system()
    if which_sys == "Linux" or which_sys == "Darwin":
        sh.tar(
            "--exclude",
            "%s/_tmp" % project_dir,
            "-cvzf",
            "%s" % tar_package_path,
            glob.glob("%s/*" % project_dir),
            _fg=True)
    six.print_("Packaging Done!\n")
    return tar_package_path
