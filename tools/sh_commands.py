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

import falcon_cli
import filelock
import glob
import os
import re
import sh
import subprocess
import sys
import time
import urllib


sys.path.insert(0, "mace/python/tools")
try:
    from encrypt_opencl_codegen import encrypt_opencl_codegen
    from opencl_codegen import opencl_codegen
    from binary_codegen import tuning_param_codegen
    from generate_data import generate_input_data
    from validate import validate
except Exception as e:
    print("Import error:\n%s" % e)
    exit(1)


################################
# common
################################
def strip_invalid_utf8(str):
    return sh.iconv(str, "-c", "-t", "UTF-8")


def make_output_processor(buff):
    def process_output(line):
        print(line.rstrip())
        buff.append(line)

    return process_output


def device_lock_path(serialno):
    return "/tmp/device-lock-%s" % serialno


def device_lock(serialno, timeout=3600):
    return filelock.FileLock(device_lock_path(serialno), timeout=timeout)


def is_device_locked(serialno):
    try:
        with device_lock(serialno, timeout=0.000001):
            return False
    except filelock.Timeout:
        return True


def formatted_file_name(input_name, input_file_name):
    return input_file_name + '_' + \
           re.sub('[^0-9a-zA-Z]+', '_', input_name)


################################
# clear data
################################
def clear_mace_run_data(abi,
                        serialno,
                        phone_data_dir,
                        model_codegen_dir="mace/codegen/models"):
    if abi != "host":
        sh.adb("-s",
               serialno,
               "shell",
               "rm -rf %s" % phone_data_dir)
    if os.path.exists(model_codegen_dir):
        sh.rm("-rf", model_codegen_dir)


################################
# adb commands
################################
def adb_split_stdout(stdout_str):
    stdout_str = strip_invalid_utf8(stdout_str)
    # Filter out last empty line
    return [l.strip() for l in stdout_str.split('\n') if len(l.strip()) > 0]


def adb_devices():
    serialnos = []
    p = re.compile(r'(\w+)\s+device')
    for line in adb_split_stdout(sh.adb("devices")):
        m = p.match(line)
        if m:
            serialnos.append(m.group(1))

    return serialnos


def get_soc_serialnos_map():
    serialnos = adb_devices()
    soc_serialnos_map = {}
    for serialno in serialnos:
        props = adb_getprop_by_serialno(serialno)
        soc_serialnos_map.setdefault(props["ro.board.platform"], [])\
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
    raw_props = adb_split_stdout(outputs)
    props = {}
    p = re.compile(r'\[(.+)\]: \[(.+)\]')
    for raw_prop in raw_props:
        m = p.match(raw_prop)
        if m:
            props[m.group(1)] = m.group(2)
    return props


def adb_get_device_name_by_serialno(serialno):
    props = adb_getprop_by_serialno(serialno)
    return props.get("ro.product.model", "")


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
    print("Push %s to %s" % (src_path, dst_path))
    sh.adb("-s", serialno, "push", src_path, dst_path)


def adb_pull(src_path, dst_path, serialno):
    print("Pull %s to %s" % (src_path, dst_path))
    try:
        sh.adb("-s", serialno, "pull", src_path, dst_path)
    except Exception as e:
        print("Error msg: %s" % e.stderr)


def adb_run(serialno,
            host_bin_path,
            bin_name,
            args="",
            opencl_profiling=1,
            vlog_level=0,
            device_bin_path="/data/local/tmp/mace",
            out_of_range_check=1):
    host_bin_full_path = "%s/%s" % (host_bin_path, bin_name)
    device_bin_full_path = "%s/%s" % (device_bin_path, bin_name)
    props = adb_getprop_by_serialno(serialno)
    print(
        "====================================================================="
    )
    print("Trying to lock device %s" % serialno)
    with device_lock(serialno):
        print("Run on device: %s, %s, %s" %
              (serialno, props["ro.board.platform"],
               props["ro.product.model"]))
        sh.adb("-s", serialno, "shell", "rm -rf %s" % device_bin_path)
        sh.adb("-s", serialno, "shell", "mkdir -p %s" % device_bin_path)
        adb_push(host_bin_full_path, device_bin_full_path, serialno)
        print("Run %s" % device_bin_full_path)
        stdout_buff = []
        process_output = make_output_processor(stdout_buff)
        p = sh.adb(
            "-s",
            serialno,
            "shell",
            "MACE_OUT_OF_RANGE_CHECK=%d MACE_OPENCL_PROFILING=%d "
            "MACE_CPP_MIN_VLOG_LEVEL=%d %s %s" %
            (out_of_range_check, opencl_profiling, vlog_level,
             device_bin_full_path, args),
            _out=process_output,
            _bg=True,
            _err_to_out=True)
        p.wait()
        return "".join(stdout_buff)


def adb_run_valgrind(serialno,
                     host_bin_path,
                     bin_name,
                     valgrind_path="/data/local/valgrind",
                     valgrind_args="",
                     args="",
                     opencl_profiling=1,
                     vlog_level=0,
                     device_bin_path="/data/local/tmp/mace",
                     out_of_range_check=1):
    valgrind_lib = valgrind_path + "/lib/valgrind"
    valgrind_bin = valgrind_path + "/bin/valgrind"
    host_bin_full_path = "%s/%s" % (host_bin_path, bin_name)
    device_bin_full_path = "%s/%s" % (device_bin_path, bin_name)
    props = adb_getprop_by_serialno(serialno)
    print(
        "====================================================================="
    )
    print("Trying to lock device %s" % serialno)
    with device_lock(serialno):
        print("Run on device: %s, %s, %s" %
              (serialno, props["ro.board.platform"],
               props["ro.product.model"]))
        result = sh.adb("-s", serialno, "shell", "ls %s" % valgrind_path)
        if result.startswith("ls:"):
            print("Please install valgrind to %s manually." % valgrind_path)
            return result
        sh.adb("-s", serialno, "shell", "rm -rf %s" % device_bin_path)
        sh.adb("-s", serialno, "shell", "mkdir -p %s" % device_bin_path)
        adb_push(host_bin_full_path, device_bin_full_path, serialno)
        print("Run %s" % device_bin_full_path)
        stdout_buff = []
        process_output = make_output_processor(stdout_buff)
        p = sh.adb(
            "-s",
            serialno,
            "shell",
            "MACE_OUT_OF_RANGE_CHECK=%d MACE_OPENCL_PROFILING=%d "
            "MACE_CPP_MIN_VLOG_LEVEL=%d VALGRIND_LIB=%s %s %s %s %s " %
            (out_of_range_check, opencl_profiling, vlog_level,
             valgrind_lib, valgrind_bin, valgrind_args,
             device_bin_full_path, args),
            _out=process_output,
            _bg=True,
            _err_to_out=True)
        p.wait()
        return "".join(stdout_buff)


################################
# bazel commands
################################
def bazel_build(target,
                strip="always",
                abi="armeabi-v7a",
                model_tag="",
                production_mode=False,
                hexagon_mode=False,
                disable_no_tuning_warning=False,
                debug=False,
                enable_openmp=True):
    print("* Build %s with ABI %s" % (target, abi))
    stdout_buff = []
    process_output = make_output_processor(stdout_buff)
    if abi == "host":
        p = sh.bazel(
            "build",
            "-c",
            "opt",
            "--strip",
            strip,
            "--verbose_failures",
            target,
            "--copt=-std=c++11",
            "--copt=-D_GLIBCXX_USE_C99_MATH_TR1",
            "--copt=-Werror=return-type",
            "--copt=-DMACE_MODEL_TAG=%s" % model_tag,
            "--copt=-O3",
            "--define",
            "openmp=%s" % str(enable_openmp).lower(),
            "--define",
            "production=%s" % str(production_mode).lower(),
            _out=process_output,
            _bg=True,
            _err_to_out=True)
        p.wait()
    else:
        bazel_args = (
            "build",
            "-c",
            "opt",
            "--strip",
            strip,
            "--verbose_failures",
            target,
            "--crosstool_top=//external:android/crosstool",
            "--host_crosstool_top=@bazel_tools//tools/cpp:toolchain",
            "--cpu=%s" % abi,
            "--copt=-std=c++11",
            "--copt=-D_GLIBCXX_USE_C99_MATH_TR1",
            "--copt=-Werror=return-type",
            "--copt=-DMACE_OBFUSCATE_LITERALS",
            "--copt=-DMACE_MODEL_TAG=%s" % model_tag,
            "--copt=-O3",
            "--define",
            "neon=true",
            "--define",
            "openmp=%s" % str(enable_openmp).lower(),
            "--define",
            "production=%s" % str(production_mode).lower(),
            "--define",
            "hexagon=%s" % str(hexagon_mode).lower())
        if disable_no_tuning_warning:
            bazel_args += ("--copt=-DMACE_DISABLE_NO_TUNING_WARNING",)
        if debug:
            bazel_args += ("--copt=-g",)
        p = sh.bazel(
            _out=process_output,
            _bg=True,
            _err_to_out=True,
            *bazel_args)
        p.wait()
    print("Building done!\n")
    return "".join(stdout_buff)


def bazel_build_common(target, build_args=""):
    stdout_buff = []
    process_output = make_output_processor(stdout_buff)
    p = sh.bazel(
        "build",
        target + build_args,
        _out=process_output,
        _bg=True,
        _err_to_out=True)
    p.wait()
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
    encrypt_opencl_codegen("./mace/kernels/opencl/cl/",
                           "mace/codegen/opencl/opencl_encrypt_program.cc")


def pull_binaries(abi, serialno, model_output_dirs):
    compiled_opencl_dir = "/data/local/tmp/mace_run/cl_program/"
    mace_run_param_file = "mace_run.config"

    cl_bin_dirs = []
    for d in model_output_dirs:
        cl_bin_dirs.append(os.path.join(d, "opencl_bin"))
    cl_bin_dirs_str = ",".join(cl_bin_dirs)
    if cl_bin_dirs:
        cl_bin_dir = cl_bin_dirs_str
        if os.path.exists(cl_bin_dir):
            sh.rm("-rf", cl_bin_dir)
        sh.mkdir("-p", cl_bin_dir)
        if abi != "host":
            adb_pull(compiled_opencl_dir, cl_bin_dir, serialno)
            adb_pull("/data/local/tmp/mace_run/%s" % mace_run_param_file,
                     cl_bin_dir, serialno)


def gen_opencl_binary_code(model_output_dirs,
                           codegen_path="mace/codegen"):
    cl_built_kernel_file_name = "mace_cl_compiled_program.bin"
    cl_platform_info_file_name = "mace_cl_platform_info.txt"
    opencl_codegen_file = "%s/opencl/opencl_compiled_program.cc" % codegen_path

    cl_bin_dirs = []
    for d in model_output_dirs:
        cl_bin_dirs.append(os.path.join(d, "opencl_bin"))
    cl_bin_dirs_str = ",".join(cl_bin_dirs)
    opencl_codegen(opencl_codegen_file,
                   cl_bin_dirs_str,
                   cl_built_kernel_file_name,
                   cl_platform_info_file_name)


def gen_tuning_param_code(model_output_dirs,
                          codegen_path="mace/codegen"):
    mace_run_param_file = "mace_run.config"
    cl_bin_dirs = []
    for d in model_output_dirs:
        cl_bin_dirs.append(os.path.join(d, "opencl_bin"))
    cl_bin_dirs_str = ",".join(cl_bin_dirs)

    tuning_codegen_dir = "%s/tuning/" % codegen_path
    if not os.path.exists(tuning_codegen_dir):
        sh.mkdir("-p", tuning_codegen_dir)

    tuning_param_variable_name = "kTuningParamsData"
    tuning_param_codegen(cl_bin_dirs_str,
                         mace_run_param_file,
                         "%s/tuning_params.cc" % tuning_codegen_dir,
                         tuning_param_variable_name)


def gen_mace_version(codegen_path="mace/codegen"):
    sh.mkdir("-p", "%s/version" % codegen_path)
    sh.bash("mace/tools/git/gen_version_source.sh",
            "%s/version/version.cc" % codegen_path)


def gen_compiled_opencl_source(codegen_path="mace/codegen"):
    opencl_codegen_file = "%s/opencl/opencl_compiled_program.cc" % codegen_path
    sh.mkdir("-p", "%s/opencl" % codegen_path)
    opencl_codegen(opencl_codegen_file)


def gen_model_code(model_codegen_dir,
                   platform,
                   model_file_path,
                   weight_file_path,
                   model_sha256_checksum,
                   input_nodes,
                   output_nodes,
                   data_type,
                   runtime,
                   model_tag,
                   input_shapes,
                   dsp_mode,
                   embed_model_data,
                   fast_conv,
                   obfuscate):
    print("* Genearte model code")
    bazel_build_common("//mace/python/tools:converter")
    if os.path.exists(model_codegen_dir):
        sh.rm("-rf", model_codegen_dir)
    sh.mkdir("-p", model_codegen_dir)
    stdout_buff = []
    process_output = make_output_processor(stdout_buff)
    p = sh.python("bazel-bin/mace/python/tools/converter",
                  "-u",
                  "--platform=%s" % platform,
                  "--model_file=%s" % model_file_path,
                  "--weight_file=%s" % weight_file_path,
                  "--model_checksum=%s" % model_sha256_checksum,
                  "--output=%s" % model_codegen_dir + "/model.cc",
                  "--input_node=%s" % input_nodes,
                  "--output_node=%s" % output_nodes,
                  "--data_type=%s" % data_type,
                  "--runtime=%s" % runtime,
                  "--output_type=source",
                  "--template=%s" % "mace/python/tools",
                  "--model_tag=%s" % model_tag,
                  "--input_shape=%s" % input_shapes,
                  "--dsp_mode=%s" % dsp_mode,
                  "--embed_model_data=%s" % embed_model_data,
                  "--winograd=%s" % fast_conv,
                  "--obfuscate=%s" % obfuscate,
                  _out=process_output,
                  _bg=True,
                  _err_to_out=True)
    p.wait()
    print("Model code gen done!\n")


def gen_random_input(model_output_dir,
                     input_nodes,
                     input_shapes,
                     input_files,
                     input_file_name="model_input"):
    for input_name in input_nodes:
        formatted_name = formatted_file_name(input_name, input_file_name)
        if os.path.exists("%s/%s" % (model_output_dir, formatted_name)):
            sh.rm("%s/%s" % (model_output_dir, formatted_name))
    input_nodes_str = ",".join(input_nodes)
    input_shapes_str = ":".join(input_shapes)
    generate_input_data("%s/%s" % (model_output_dir, input_file_name),
                        input_nodes_str,
                        input_shapes_str)

    input_file_list = []
    if isinstance(input_files, list):
        input_file_list.extend(input_files)
    else:
        input_file_list.append(input_files)
    if len(input_file_list) != 0:
        input_name_list = []
        if isinstance(input_nodes, list):
            input_name_list.extend(input_nodes)
        else:
            input_name_list.append(input_nodes)
        if len(input_file_list) != len(input_name_list):
            raise Exception('If input_files set, the input files should '
                            'match the input names.')
        for i in range(len(input_file_list)):
            if input_file_list[i] is not None:
                dst_input_file = model_output_dir + '/' + \
                        formatted_file_name(input_name_list[i],
                                            input_file_name)
                if input_file_list[i].startswith("http://") or \
                        input_file_list[i].startswith("https://"):
                    urllib.urlretrieve(input_file_list[i], dst_input_file)
                else:
                    sh.cp("-f", input_file_list[i], dst_input_file)


def update_mace_run_lib(model_output_dir,
                        abi,
                        model_tag,
                        embed_model_data,
                        generated_model_lib_dir="bazel-bin/mace/codegen/"):
    model_lib_path = model_output_dir + "/libmace_%s.a" % model_tag
    if abi == "host":
        bazel_build(
                "//mace/codegen:generated_models",
                abi=abi,
                model_tag=model_tag)
        generated_model_lib_name = "libgenerated_models.pic.a"
    else:
        generated_model_lib_name = "libgenerated_models.a"

    if os.path.exists(model_lib_path):
        sh.rm("-rf", model_lib_path)
    sh.cp("-f", generated_model_lib_dir + "/" + generated_model_lib_name,
          model_lib_path)

    mace_run_filepath = model_output_dir + "/mace_run"
    if os.path.exists(mace_run_filepath):
        sh.rm("-rf", mace_run_filepath)
    sh.cp("-f", "bazel-bin/mace/tools/validation/mace_run", model_output_dir)

    if embed_model_data == 0:
        sh.cp("-f", "mace/codegen/models/%s/%s.data" % (model_tag, model_tag),
              model_output_dir)

    sh.cp("-f", "mace/codegen/models/%s/%s.h" % (model_tag, model_tag),
          model_output_dir)


def tuning_run(abi,
               serialno,
               vlog_level,
               embed_model_data,
               model_output_dir,
               input_nodes,
               output_nodes,
               input_shapes,
               output_shapes,
               model_tag,
               device_type,
               running_round,
               restart_round,
               limit_opencl_kernel_time,
               tuning,
               out_of_range_check,
               phone_data_dir,
               omp_num_threads=-1,
               cpu_affinity_policy=1,
               gpu_perf_hint=3,
               gpu_priority_hint=3,
               input_file_name="model_input",
               output_file_name="model_out"):
    print("* Run '%s' with round=%s, restart_round=%s, tuning=%s, "
          "out_of_range_check=%s, omp_num_threads=%s, cpu_affinity_policy=%s, "
          "gpu_perf_hint=%s, gpu_priority_hint=%s" %
          (model_tag, running_round, restart_round, str(tuning),
           str(out_of_range_check), omp_num_threads, cpu_affinity_policy,
           gpu_perf_hint, gpu_priority_hint))
    if abi == "host":
        p = subprocess.Popen(
            [
                "env",
                "MACE_CPP_MIN_VLOG_LEVEL=%s" % vlog_level,
                "%s/mace_run" % model_output_dir,
                "--input_node=%s" % ",".join(input_nodes),
                "--output_node=%s" % ",".join(output_nodes),
                "--input_shape=%s" % ":".join(input_shapes),
                "--output_shape=%s" % ":".join(output_shapes),
                "--input_file=%s/%s" % (model_output_dir, input_file_name),
                "--output_file=%s/%s" % (model_output_dir, output_file_name),
                "--model_data_file=%s/%s.data" % (model_output_dir, model_tag),
                "--device=%s" % device_type,
                "--round=%s" % running_round,
                "--restart_round=%s" % restart_round,
                "--omp_num_threads=%s" % omp_num_threads,
                "--cpu_affinity_policy=%s" % cpu_affinity_policy,
                "--gpu_perf_hint=%s" % gpu_perf_hint,
                "--gpu_priority_hint=%s" % gpu_priority_hint,
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE)
        out, err = p.communicate()
        stdout = err + out
        print stdout
        print("Running finished!\n")
        return stdout
    else:
        sh.adb("-s", serialno, "shell", "mkdir", "-p", phone_data_dir)
        compiled_opencl_dir = "/data/local/tmp/mace_run/cl_program/"
        sh.adb("-s", serialno, "shell", "mkdir", "-p", compiled_opencl_dir)

        for input_name in input_nodes:
            formatted_name = formatted_file_name(input_name,
                                                 input_file_name)
            adb_push("%s/%s" % (model_output_dir, formatted_name),
                     phone_data_dir, serialno)
        adb_push("%s/mace_run" % model_output_dir, phone_data_dir,
                 serialno)
        if not embed_model_data:
            adb_push("%s/%s.data" % (model_output_dir, model_tag),
                     phone_data_dir, serialno)
        adb_push("mace/core/runtime/hexagon/libhexagon_controller.so",
                 phone_data_dir, serialno)

        stdout_buff = []
        process_output = make_output_processor(stdout_buff)
        p = sh.adb(
            "-s",
            serialno,
            "shell",
            "LD_LIBRARY_PATH=%s" % phone_data_dir,
            "MACE_TUNING=%s" % int(tuning),
            "MACE_OUT_OF_RANGE_CHECK=%s" % int(out_of_range_check),
            "MACE_CPP_MIN_VLOG_LEVEL=%s" % vlog_level,
            "MACE_RUN_PARAMETER_PATH=%s/mace_run.config" %
            phone_data_dir,
            "MACE_CL_PROGRAM_PATH=%s/cl_program" % phone_data_dir,
            "MACE_LIMIT_OPENCL_KERNEL_TIME=%s" %
            limit_opencl_kernel_time,
            "%s/mace_run" % phone_data_dir,
            "--input_node=%s" % ",".join(input_nodes),
            "--output_node=%s" % ",".join(output_nodes),
            "--input_shape=%s" % ":".join(input_shapes),
            "--output_shape=%s" % ":".join(output_shapes),
            "--input_file=%s/%s" % (phone_data_dir, input_file_name),
            "--output_file=%s/%s" % (phone_data_dir, output_file_name),
            "--model_data_file=%s/%s.data" % (phone_data_dir, model_tag),
            "--device=%s" % device_type,
            "--round=%s" % running_round,
            "--restart_round=%s" % restart_round,
            "--omp_num_threads=%s" % omp_num_threads,
            "--cpu_affinity_policy=%s" % cpu_affinity_policy,
            "--gpu_perf_hint=%s" % gpu_perf_hint,
            "--gpu_priority_hint=%s" % gpu_priority_hint,
            _out=process_output,
            _bg=True,
            _err_to_out=True)
        p.wait()
        print("Running finished!\n")
        return "".join(stdout_buff)


def validate_model(abi,
                   serialno,
                   model_file_path,
                   weight_file_path,
                   platform,
                   runtime,
                   input_nodes,
                   output_nodes,
                   input_shapes,
                   output_shapes,
                   model_output_dir,
                   phone_data_dir,
                   input_file_name="model_input",
                   output_file_name="model_out"):
    print("* Validate with %s" % platform)

    if platform == "tensorflow":
        if abi != "host":
            for output_name in output_nodes:
                formatted_name = formatted_file_name(
                        output_name, output_file_name)
                if os.path.exists("%s/%s" % (model_output_dir,
                                             formatted_name)):
                    sh.rm("%s/%s" % (model_output_dir, formatted_name))
                adb_pull("%s/%s" % (phone_data_dir, formatted_name),
                         model_output_dir, serialno)
        validate(platform, model_file_path, "",
                 "%s/%s" % (model_output_dir, input_file_name),
                 "%s/%s" % (model_output_dir, output_file_name), runtime,
                 ":".join(input_shapes), ":".join(output_shapes),
                 ",".join(input_nodes), ",".join(output_nodes))
    elif platform == "caffe":
        image_name = "mace-caffe:latest"
        container_name = "mace_caffe_validator"
        res_file = "validation.result"

        docker_image_id = sh.docker("images", "-q", image_name)
        if not docker_image_id:
            print("Build caffe docker")
            sh.docker("build", "-t", image_name, "docker/caffe")

        container_id = sh.docker("ps", "-qa", "-f", "name=%s" % container_name)
        if container_id and not sh.docker("ps", "-qa", "--filter",
                                          "status=running", "-f",
                                          "name=%s" % container_name):
            sh.docker("rm", "-f", container_name)
            container_id = ""
        if not container_id:
            print("Run caffe container")
            sh.docker(
                    "run",
                    "-d",
                    "-it",
                    "--name",
                    container_name,
                    image_name,
                    "/bin/bash")

        for input_name in input_nodes:
            formatted_input_name = formatted_file_name(
                    input_name, input_file_name)
            sh.docker(
                    "cp",
                    "%s/%s" % (model_output_dir, formatted_input_name),
                    "%s:/mace" % container_name)

        if abi != "host":
            for output_name in output_nodes:
                formatted_output_name = formatted_file_name(
                        output_name, output_file_name)
                sh.rm("-rf",
                      "%s/%s" % (model_output_dir, formatted_output_name))
                adb_pull("%s/%s" % (phone_data_dir, formatted_output_name),
                         model_output_dir, serialno)

        for output_name in output_nodes:
            formatted_output_name = formatted_file_name(
                    output_name, output_file_name)
            sh.docker(
                    "cp",
                    "%s/%s" % (model_output_dir, formatted_output_name),
                    "%s:/mace" % container_name)
        model_file_name = os.path.basename(model_file_path)
        weight_file_name = os.path.basename(weight_file_path)
        sh.docker("cp", "tools/validate.py", "%s:/mace" % container_name)
        sh.docker("cp", model_file_path, "%s:/mace" % container_name)
        sh.docker("cp", weight_file_path, "%s:/mace" % container_name)

        stdout_buff = []
        process_output = make_output_processor(stdout_buff)
        p = sh.docker(
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
                "--mace_runtime=%s" % runtime,
                "--input_node=%s" % ",".join(input_nodes),
                "--output_node=%s" % ",".join(output_nodes),
                "--input_shape=%s" % ":".join(input_shapes),
                "--output_shape=%s" % ":".join(output_shapes),
                _out=process_output,
                _bg=True,
                _err_to_out=True)
        p.wait()

    print("Validation done!\n")


def build_production_code(abi):
    bazel_build("//mace/codegen:generated_opencl", abi=abi)
    bazel_build("//mace/codegen:generated_tuning_params", abi=abi)


def merge_libs(target_soc,
               abi,
               project_name,
               libmace_output_dir,
               model_output_dirs,
               hexagon_mode,
               embed_model_data):
    print("* Merge mace lib")
    project_output_dir = "%s/%s" % (libmace_output_dir, project_name)
    model_header_dir = "%s/include/mace/public" % project_output_dir
    model_data_dir = "%s/data" % project_output_dir
    hexagon_lib_file = "mace/core/runtime/hexagon/libhexagon_controller.so"
    model_bin_dir = "%s/%s/" % (project_output_dir, abi)

    if not os.path.exists(model_bin_dir):
        sh.mkdir("-p", model_bin_dir)
    if not os.path.exists(model_header_dir):
        sh.mkdir("-p", model_header_dir)
    sh.cp("-f", glob.glob("mace/public/*.h"), model_header_dir)
    if not os.path.exists(model_data_dir):
        sh.mkdir("-p", model_data_dir)
    if hexagon_mode:
        sh.cp("-f", hexagon_lib_file, model_bin_dir)

    mri_stream = ""
    if abi == "host":
        mri_stream += "create %s/libmace_%s.a\n" % \
                      (model_bin_dir, project_name)
        mri_stream += (
                "addlib "
                "bazel-bin/mace/codegen/libgenerated_opencl.pic.a\n")
        mri_stream += (
                "addlib "
                "bazel-bin/mace/codegen/libgenerated_tuning_params.pic.a\n")
    else:
        mri_stream += "create %s/libmace_%s.%s.a\n" % \
                      (model_bin_dir, project_name, target_soc)
        mri_stream += (
                "addlib "
                "bazel-bin/mace/codegen/libgenerated_opencl.a\n")
        mri_stream += (
                "addlib "
                "bazel-bin/mace/codegen/libgenerated_tuning_params.a\n")
        mri_stream += (
                "addlib "
                "bazel-bin/mace/codegen/libgenerated_version.a\n")
        mri_stream += (
                "addlib "
                "bazel-bin/mace/core/libcore.a\n")
        mri_stream += (
                "addlib "
                "bazel-bin/mace/kernels/libkernels.a\n")
        mri_stream += (
                "addlib "
                "bazel-bin/mace/utils/libutils.a\n")
        mri_stream += (
                "addlib "
                "bazel-bin/mace/utils/libutils_prod.a\n")
        mri_stream += (
                "addlib "
                "bazel-bin/mace/ops/libops.lo\n")

    for model_output_dir in model_output_dirs:
        for lib in sh.ls(glob.glob("%s/*.a" % model_output_dir), "-1"):
            mri_stream += "addlib %s\n" % lib
        if not embed_model_data:
            sh.cp("-f", glob.glob("%s/*.data" % model_output_dir),
                  model_data_dir)
        sh.cp("-f", glob.glob("%s/*.h" % model_output_dir), model_header_dir)

    mri_stream += "save\n"
    mri_stream += "end\n"

    cmd = sh.Command("%s/toolchains/" % os.environ["ANDROID_NDK_HOME"] +
                     "aarch64-linux-android-4.9/prebuilt/linux-x86_64/" +
                     "bin/aarch64-linux-android-ar")

    cmd("-M", _in=mri_stream)

    print("Libs merged!\n")


def packaging_lib(libmace_output_dir, project_name):
    print("* Package libs for %s" % project_name)
    tar_package_name = "libmace_%s.tar.gz" % project_name
    project_dir = "%s/%s" % (libmace_output_dir, project_name)
    tar_package_path = "%s/%s" % (project_dir, tar_package_name)
    if os.path.exists(tar_package_path):
        sh.rm("-rf", tar_package_path)

    print("Start packaging '%s' libs into %s" % (project_name,
                                                 tar_package_path))
    # ls ${project_dir} -1 | grep -v build | grep -v .tar.gz | xargs -I {} \
    #       tar cvzf ${project_dir}/${tar_package_name} ${project_name}/{}
    sh.xargs(
            sh.grep(
                sh.grep(
                    sh.ls(project_dir, "-1"),
                    "-v", "build"),
                "-v", ".tar.gz"),
            "-I",
            "{}",
            "tar",
            "cvzf",
            "%s" % tar_package_path,
            "%s/{}" % project_dir)
    print("Packaging Done!\n")


def benchmark_model(abi,
                    serialno,
                    vlog_level,
                    embed_model_data,
                    model_output_dir,
                    input_nodes,
                    output_nodes,
                    input_shapes,
                    output_shapes,
                    model_tag,
                    device_type,
                    hexagon_mode,
                    phone_data_dir,
                    omp_num_threads=-1,
                    cpu_affinity_policy=1,
                    gpu_perf_hint=3,
                    gpu_priority_hint=3,
                    input_file_name="model_input",
                    output_file_name="model_out"):
    print("* Benchmark for %s" % model_tag)
    benchmark_binary_file = "%s/benchmark_model" % model_output_dir
    if os.path.exists(benchmark_binary_file):
        sh.rm("-rf", benchmark_binary_file)
    if not embed_model_data:
        sh.cp("-f", "mace/codegen/models/%s/%s.data" % (model_tag, model_tag),
              model_output_dir)

    benchmark_target = "//mace/benchmark:benchmark_model"
    bazel_build(benchmark_target,
                abi=abi,
                model_tag=model_tag,
                production_mode=True,
                hexagon_mode=hexagon_mode)

    target_bin = "/".join(bazel_target_to_bin(benchmark_target))
    sh.cp("-f", target_bin, model_output_dir)

    stdout_buff = []
    process_output = make_output_processor(stdout_buff)
    if abi == "host":
        p = subprocess.Popen(
            [
                "env",
                "MACE_CPP_MIN_VLOG_LEVEL=%s" % vlog_level,
                "%s/benchmark_model" % model_output_dir,
                "--input_node=%s" % ",".join(input_nodes),
                "--output_node=%s" % ",".join(output_nodes),
                "--input_shape=%s" % ":".join(input_shapes),
                "--output_shape=%s" % ":".join(output_shapes),
                "--input_file=%s/%s" % (model_output_dir, input_file_name),
                "--model_data_file=%s/%s.data" % (model_output_dir, model_tag),
                "--device=%s" % device_type,
                "--omp_num_threads=%s" % omp_num_threads,
                "--cpu_affinity_policy=%s" % cpu_affinity_policy,
                "--gpu_perf_hint=%s" % gpu_perf_hint,
                "--gpu_priority_hint=%s" % gpu_priority_hint,
            ])
        p.wait()
    else:
        sh.adb("-s", serialno, "shell", "mkdir", "-p", phone_data_dir)

        for input_name in input_nodes:
            formatted_name = formatted_file_name(input_name,
                                                 input_file_name)
            adb_push("%s/%s" % (model_output_dir, formatted_name),
                     phone_data_dir, serialno)
        adb_push("%s/benchmark_model" % model_output_dir, phone_data_dir,
                 serialno)
        if not embed_model_data:
            adb_push("%s/%s.data" % (model_output_dir, model_tag),
                     phone_data_dir, serialno)
        p = sh.adb(
            "-s",
            serialno,
            "shell",
            "LD_LIBRARY_PATH=%s" % phone_data_dir,
            "MACE_CPP_MIN_VLOG_LEVEL=%s" % vlog_level,
            "MACE_RUN_PARAMETER_PATH=%s/mace_run.config" %
            phone_data_dir,
            "MACE_OPENCL_PROFILING=1",
            "%s/benchmark_model" % phone_data_dir,
            "--input_node=%s" % ",".join(input_nodes),
            "--output_node=%s" % ",".join(output_nodes),
            "--input_shape=%s" % ":".join(input_shapes),
            "--output_shape=%s" % ":".join(output_shapes),
            "--input_file=%s/%s" % (phone_data_dir, input_file_name),
            "--model_data_file=%s/%s.data" % (phone_data_dir, model_tag),
            "--device=%s" % device_type,
            "--omp_num_threads=%s" % omp_num_threads,
            "--cpu_affinity_policy=%s" % cpu_affinity_policy,
            "--gpu_perf_hint=%s" % gpu_perf_hint,
            "--gpu_priority_hint=%s" % gpu_priority_hint,
            _out=process_output,
            _bg=True,
            _err_to_out=True)
        p.wait()

    print("Benchmark done!\n")
    return "".join(stdout_buff)


def build_run_throughput_test(abi,
                              serialno,
                              vlog_level,
                              run_seconds,
                              merged_lib_file,
                              model_input_dir,
                              embed_model_data,
                              input_nodes,
                              output_nodes,
                              input_shapes,
                              output_shapes,
                              cpu_model_tag,
                              gpu_model_tag,
                              dsp_model_tag,
                              phone_data_dir,
                              strip="always",
                              input_file_name="model_input"):
    print("* Build and run throughput_test")

    model_tag_build_flag = ""
    if cpu_model_tag:
        model_tag_build_flag += "--copt=-DMACE_CPU_MODEL_TAG=%s " % \
                                cpu_model_tag
    if gpu_model_tag:
        model_tag_build_flag += "--copt=-DMACE_GPU_MODEL_TAG=%s " % \
                                gpu_model_tag
    if dsp_model_tag:
        model_tag_build_flag += "--copt=-DMACE_DSP_MODEL_TAG=%s " % \
                                dsp_model_tag

    sh.cp("-f", merged_lib_file, "mace/benchmark/libmace_merged.a")
    stdout_buff = []
    process_output = make_output_processor(stdout_buff)
    p = sh.bazel(
        "build",
        "-c",
        "opt",
        "--strip",
        strip,
        "--verbose_failures",
        "//mace/benchmark:model_throughput_test",
        "--crosstool_top=//external:android/crosstool",
        "--host_crosstool_top=@bazel_tools//tools/cpp:toolchain",
        "--cpu=%s" % abi,
        "--copt=-std=c++11",
        "--copt=-D_GLIBCXX_USE_C99_MATH_TR1",
        "--copt=-Werror=return-type",
        "--copt=-O3",
        "--define",
        "neon=true",
        "--define",
        "openmp=true",
        model_tag_build_flag,
        _out=process_output,
        _bg=True,
        _err_to_out=True)
    p.wait()

    sh.rm("mace/benchmark/libmace_merged.a")
    sh.adb("-s",
           serialno,
           "shell",
           "mkdir",
           "-p",
           phone_data_dir)
    adb_push("%s/%s_%s" % (model_input_dir, input_file_name,
                           ",".join(input_nodes)),
             phone_data_dir,
             serialno)
    adb_push("bazel-bin/mace/benchmark/model_throughput_test",
             phone_data_dir,
             serialno)
    if not embed_model_data:
        adb_push("codegen/models/%s/%s.data" % cpu_model_tag,
                 phone_data_dir,
                 serialno)
        adb_push("codegen/models/%s/%s.data" % gpu_model_tag,
                 phone_data_dir,
                 serialno)
        adb_push("codegen/models/%s/%s.data" % dsp_model_tag,
                 phone_data_dir,
                 serialno)
    adb_push("mace/core/runtime/hexagon/libhexagon_controller.so",
             phone_data_dir,
             serialno)

    p = sh.adb(
            "-s",
            serialno,
            "shell",
            "LD_LIBRARY_PATH=%s" % phone_data_dir,
            "MACE_CPP_MIN_VLOG_LEVEL=%s" % vlog_level,
            "MACE_RUN_PARAMETER_PATH=%s/mace_run.config" %
            phone_data_dir,
            "%s/model_throughput_test" % phone_data_dir,
            "--input_node=%s" % ",".join(input_nodes),
            "--output_node=%s" % ",".join(output_nodes),
            "--input_shape=%s" % ":".join(input_shapes),
            "--output_shape=%s" % ":".join(output_shapes),
            "--input_file=%s/%s" % (phone_data_dir, input_file_name),
            "--cpu_model_data_file=%s/%s.data" % (phone_data_dir,
                                                  cpu_model_tag),
            "--gpu_model_data_file=%s/%s.data" % (phone_data_dir,
                                                  gpu_model_tag),
            "--dsp_model_data_file=%s/%s.data" % (phone_data_dir,
                                                  dsp_model_tag),
            "--run_seconds=%s" % run_seconds,
            _out=process_output,
            _bg=True,
            _err_to_out=True)
    p.wait()

    print("throughput_test done!\n")


################################
# falcon
################################
def falcon_tags(tags_dict):
    tags = ""
    for k, v in tags_dict.iteritems():
        if tags == "":
            tags = "%s=%s" % (k, v)
        else:
            tags = tags + ",%s=%s" % (k, v)
    return tags


def falcon_push_metrics(metrics, endpoint="mace_dev", tags={}):
    cli = falcon_cli.FalconCli.connect(
        server="transfer.falcon.miliao.srv", port=8433, debug=False)
    ts = int(time.time())
    falcon_metrics = [{
        "endpoint": endpoint,
        "metric": key,
        "tags": falcon_tags(tags),
        "timestamp": ts,
        "value": value,
        "step": 600,
        "counterType": "GAUGE"
    } for key, value in metrics.iteritems()]
    cli.update(falcon_metrics)
