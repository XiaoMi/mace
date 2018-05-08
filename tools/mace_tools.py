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

# python tools/mace_tools.py \
#     --config=tools/example.yaml \
#     --round=100 \
#     --mode=all

import argparse
import enum
import filelock
import hashlib
import os
import sh
import subprocess
import sys
import urllib
import yaml
import re

import common
import sh_commands

from ConfigParser import ConfigParser


def get_target_socs(configs):
    if "host" in configs["target_abis"]:
        return [""]
    else:
        available_socs = sh_commands.adb_get_all_socs()
        target_socs = available_socs
        if "target_socs" in configs:
            target_socs = set(configs["target_socs"])
            target_socs = target_socs & available_socs

        if FLAGS.target_socs != "all":
            socs = set(FLAGS.target_socs.split(','))
            target_socs = target_socs & socs
            missing_socs = socs.difference(target_socs)
            if len(missing_socs) > 0:
                print(
                    "Error: devices with SoCs are not connected %s" %
                    missing_socs)
                exit(1)

        if not target_socs:
            print("Error: no device to run")
            exit(1)

        return target_socs


def get_data_and_device_type(runtime):
    data_type = ""
    device_type = ""

    if runtime == "dsp":
        data_type = "DT_UINT8"
        device_type = "HEXAGON"
    elif runtime == "gpu":
        data_type = "DT_HALF"
        device_type = "GPU"
    elif runtime == "cpu":
        data_type = "DT_FLOAT"
        device_type = "CPU"

    return data_type, device_type


def get_hexagon_mode(configs):
    runtime_list = []
    for model_name in configs["models"]:
        model_runtime = configs["models"][model_name]["runtime"]
        runtime_list.append(model_runtime.lower())

    global_runtime = ""
    if "dsp" in runtime_list:
        return True
    return False


def gen_opencl_and_tuning_code(target_abi,
                               serialno,
                               model_output_dirs,
                               pull_or_not):
    cl_built_kernel_file_name = "mace_cl_compiled_program.bin"
    cl_platform_info_file_name = "mace_cl_platform_info.txt"
    if pull_or_not:
        sh_commands.pull_binaries(target_abi, serialno, model_output_dirs,
                                  cl_built_kernel_file_name,
                                  cl_platform_info_file_name)

    # generate opencl binary code
    sh_commands.gen_opencl_binary_code(model_output_dirs,
                                       cl_built_kernel_file_name,
                                       cl_platform_info_file_name)

    sh_commands.gen_tuning_param_code(model_output_dirs)


def model_benchmark_stdout_processor(stdout,
                                     abi,
                                     serialno,
                                     model_name,
                                     runtime):
    metrics = [0] * 5
    for line in stdout.split('\n'):
        line = line.strip()
        parts = line.split()
        if len(parts) == 6 and parts[0].startswith("time"):
            metrics[0] = str(float(parts[1]))
            metrics[1] = str(float(parts[2]))
            metrics[2] = str(float(parts[3]))
            metrics[3] = str(float(parts[4]))
            metrics[4] = str(float(parts[5]))
            break

    device_name = ""
    target_soc = ""
    if abi != "host":
        props = sh_commands.adb_getprop_by_serialno(serialno)
        device_name = props.get("ro.product.model", "")
        target_soc = props.get("ro.board.platform", "")

    report_filename = FLAGS.output_dir + "/report.csv"
    if not os.path.exists(report_filename):
        with open(report_filename, 'w') as f:
            f.write("model_name,device_name,soc,abi,runtime,create_net,"
                    "engine_ctor,init,warmup,run_avg\n")

    data_str = "{model_name},{device_name},{soc},{abi},{runtime}," \
               "{create_net},{engine_ctor},{init},{warmup},{run_avg}\n" \
        .format(
            model_name=model_name,
            device_name=device_name,
            soc=target_soc,
            abi=abi,
            runtime=runtime,
            create_net=metrics[0],
            engine_ctor=metrics[1],
            init=metrics[2],
            warmup=metrics[3],
            run_avg=metrics[4]
        )
    with open(report_filename, 'a') as f:
        f.write(data_str)


def tuning_run(runtime,
               target_abi,
               serialno,
               vlog_level,
               embed_model_data,
               model_output_dir,
               input_nodes,
               output_nodes,
               input_shapes,
               output_shapes,
               model_name,
               device_type,
               running_round,
               restart_round,
               out_of_range_check,
               phone_data_dir,
               tuning=False,
               limit_opencl_kernel_time=0,
               omp_num_threads=-1,
               cpu_affinity_policy=1,
               gpu_perf_hint=3,
               gpu_priority_hint=3):
    stdout = sh_commands.tuning_run(
        target_abi,
        serialno,
        vlog_level,
        embed_model_data,
        model_output_dir,
        input_nodes,
        output_nodes,
        input_shapes,
        output_shapes,
        model_name,
        device_type,
        running_round,
        restart_round,
        limit_opencl_kernel_time,
        tuning,
        out_of_range_check,
        phone_data_dir,
        omp_num_threads,
        cpu_affinity_policy,
        gpu_perf_hint,
        gpu_priority_hint,
        valgrind=FLAGS.valgrind,
        valgrind_path=FLAGS.valgrind_path,
        valgrind_args=FLAGS.valgrind_args
    )

    if running_round > 0 and FLAGS.collect_report:
        model_benchmark_stdout_processor(
            stdout, target_abi, serialno, model_name, runtime)


def build_mace_run_prod(hexagon_mode, runtime, target_abi,
                        serialno, vlog_level, embed_model_data,
                        model_output_dir, input_nodes, output_nodes,
                        input_shapes, output_shapes, model_name, device_type,
                        running_round, restart_round, tuning,
                        limit_opencl_kernel_time, phone_data_dir,
                        enable_openmp):
    mace_run_target = "//mace/tools/validation:mace_run"
    strip = "always"
    debug = False
    if FLAGS.valgrind:
        strip = "never"
        debug = True

    if runtime == "gpu":
        gen_opencl_and_tuning_code(target_abi, serialno, [], False)
        sh_commands.bazel_build(
            mace_run_target,
            abi=target_abi,
            production_mode=False,
            hexagon_mode=hexagon_mode,
            enable_openmp=enable_openmp
        )
        sh_commands.update_mace_run_lib(model_output_dir,
                                        model_name, embed_model_data)

        tuning_run(runtime, target_abi, serialno, vlog_level, embed_model_data,
                   model_output_dir, input_nodes, output_nodes, input_shapes,
                   output_shapes, model_name, device_type, running_round=0,
                   restart_round=1, out_of_range_check=False,
                   phone_data_dir=phone_data_dir, tuning=tuning,
                   limit_opencl_kernel_time=limit_opencl_kernel_time)

        tuning_run(runtime, target_abi, serialno, vlog_level, embed_model_data,
                   model_output_dir, input_nodes, output_nodes, input_shapes,
                   output_shapes, model_name, device_type, running_round=0,
                   restart_round=1, out_of_range_check=True,
                   phone_data_dir=phone_data_dir, tuning=False)

        gen_opencl_and_tuning_code(target_abi, serialno, [model_output_dir],
                                   True)
        sh_commands.bazel_build(
            mace_run_target,
            strip,
            abi=target_abi,
            production_mode=True,
            hexagon_mode=hexagon_mode,
            debug=debug,
            enable_openmp=enable_openmp
        )
        sh_commands.update_mace_run_lib(model_output_dir,
                                        model_name, embed_model_data)
    else:
        gen_opencl_and_tuning_code(target_abi, serialno, [], False)
        sh_commands.bazel_build(
            mace_run_target,
            strip,
            abi=target_abi,
            production_mode=True,
            hexagon_mode=hexagon_mode,
            debug=debug,
            enable_openmp=enable_openmp
        )
        sh_commands.update_mace_run_lib(model_output_dir,
                                        model_name, embed_model_data)


def merge_libs_and_tuning_results(target_soc,
                                  target_abi,
                                  serialno,
                                  project_name,
                                  output_dir,
                                  model_output_dirs,
                                  hexagon_mode,
                                  embed_model_data):
    gen_opencl_and_tuning_code(
            target_abi, serialno, model_output_dirs, False)
    sh_commands.build_production_code(target_abi)

    sh_commands.merge_libs(target_soc,
                           target_abi,
                           project_name,
                           output_dir,
                           model_output_dirs,
                           hexagon_mode,
                           embed_model_data)


def get_model_files(model_file_path,
                    model_output_dir,
                    weight_file_path=""):
    model_file = ""
    weight_file = ""
    if model_file_path.startswith("http://") or \
            model_file_path.startswith("https://"):
        model_file = model_output_dir + "/model.pb"
        urllib.urlretrieve(model_file_path, model_file)
    else:
        model_file = model_file_path

    if weight_file_path.startswith("http://") or \
            weight_file_path.startswith("https://"):
        weight_file = model_output_dir + "/model.caffemodel"
        urllib.urlretrieve(weight_file_path, weight_file)
    else:
        weight_file = weight_file_path

    return model_file, weight_file


def md5sum(str):
    md5 = hashlib.md5()
    md5.update(str)
    return md5.hexdigest()


################################
# Parsing arguments
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
        return common.CaffeEnvType.DOCKER
    elif v.lower() == 'local':
        return common.CaffeEnvType.LOCAL
    else:
        raise argparse.ArgumentTypeError('[docker | local] expected.')


def parse_model_configs():
    print("============== Load and Parse configs ==============")
    with open(FLAGS.config) as f:
        configs = yaml.load(f)
        target_abis = configs.get("target_abis", [])
        if not isinstance(target_abis, list) or not target_abis:
            print("CONFIG ERROR:")
            print("target_abis list is needed!")
            print("For example: 'target_abis: [armeabi-v7a, arm64-v8a]'")
            exit(1)

        embed_model_data = configs.get("embed_model_data", "")
        if embed_model_data == "" or not isinstance(embed_model_data, int) or \
                embed_model_data < 0 or embed_model_data > 1:
            print("CONFIG ERROR:")
            print("embed_model_data must be integer in range [0, 1]")
            exit(1)

        model_names = configs.get("models", "")
        if not model_names:
            print("CONFIG ERROR:")
            print("models attribute not found in config file")
            exit(1)

        for model_name in model_names:
            model_config = configs["models"][model_name]
            platform = model_config.get("platform", "")
            if platform == "" or platform not in ["tensorflow", "caffe"]:
                print("CONFIG ERROR:")
                print("'platform' must be 'tensorflow' or 'caffe'")
                exit(1)

            for key in ["model_file_path", "model_sha256_checksum",
                        "runtime"]:
                value = model_config.get(key, "")
                if value == "":
                    print("CONFIG ERROR:")
                    print("'%s' is necessary" % key)
                    exit(1)

            for key in ["input_nodes", "input_shapes", "output_nodes",
                        "output_shapes"]:
                value = model_config.get(key, "")
                if value == "":
                    print("CONFIG ERROR:")
                    print("'%s' is necessary" % key)
                    exit(1)
                if not isinstance(value, list):
                    model_config[key] = [value]

            for key in ["limit_opencl_kernel_time", "dsp_mode", "obfuscate",
                        "fast_conv"]:
                value = model_config.get(key, "")
                if value == "":
                    model_config[key] = 0
                    print("'%s' for %s is set to default value: 0" %
                          (key, model_name))

            validation_inputs_data = model_config.get("validation_inputs_data",
                                                      [])
            model_config["validation_inputs_data"] = validation_inputs_data
            if not isinstance(validation_inputs_data, list):
                model_config["validation_inputs_data"] = [
                        validation_inputs_data]

            weight_file_path = model_config.get("weight_file_path", "")
            model_config["weight_file_path"] = weight_file_path

        print("Parse model configs successfully!\n")
        return configs


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./tool/config",
        required=True,
        help="The global config file of models.")
    parser.add_argument(
        "--output_dir", type=str, default="build", help="The output dir.")
    parser.add_argument(
        "--round", type=int, default=1, help="The model running round.")
    parser.add_argument(
        "--run_seconds",
        type=int,
        default=10,
        help="The model throughput test running seconds.")
    parser.add_argument(
        "--restart_round",
        type=int,
        default=1,
        help="The model restart round.")
    parser.add_argument(
        "--tuning",
        type=str2bool,
        default=True,
        help="Tune opencl params.")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="[build|run|validate|benchmark|merge|all|throughput_test].")
    parser.add_argument(
        "--target_socs",
        type=str,
        default="all",
        help="SoCs to build, comma seperated list (getprop ro.board.platform)")
    parser.add_argument(
        "--out_of_range_check",
        type=str2bool,
        default=False,
        help="Enable out of range check for opencl.")
    parser.add_argument(
        "--enable_openmp",
        type=str2bool,
        default=True,
        help="Enable openmp.")
    parser.add_argument(
        "--omp_num_threads",
        type=int,
        default=-1,
        help="num of openmp threads")
    parser.add_argument(
        "--cpu_affinity_policy",
        type=int,
        default=1,
        help="0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY")
    parser.add_argument(
        "--gpu_perf_hint",
        type=int,
        default=3,
        help="0:DEFAULT/1:LOW/2:NORMAL/3:HIGH")
    parser.add_argument(
        "--gpu_priority_hint",
        type=int,
        default=3,
        help="0:DEFAULT/1:LOW/2:NORMAL/3:HIGH")
    parser.add_argument(
        "--collect_report",
        type=str2bool,
        default=False,
        help="Collect report.")
    parser.add_argument(
        "--vlog_level",
        type=int,
        default=0,
        help="VLOG level.")
    parser.add_argument(
        "--caffe_env",
        type=str_to_caffe_env_type,
        default='docker',
        help="[docker | local] caffe environment.")
    parser.add_argument(
        "--valgrind",
        type=bool,
        default=False,
        help="Whether to use valgrind to check memory error.")
    parser.add_argument(
        "--valgrind_path",
        type=str,
        default="/data/local/tmp/valgrind",
        help="Valgrind install path.")
    parser.add_argument(
        "--valgrind_args",
        type=str,
        default="",
        help="Valgrind command args.")
    return parser.parse_known_args()


def process_models(project_name, configs, embed_model_data, vlog_level,
                   target_abi, phone_data_dir, target_soc="", serialno=""):
    hexagon_mode = get_hexagon_mode(configs)
    model_output_dirs = []

    for model_name in configs["models"]:
        print '===================', model_name, '==================='
        model_config = configs["models"][model_name]
        input_file_list = model_config["validation_inputs_data"]
        data_type, device_type = get_data_and_device_type(
                model_config["runtime"])

        # Create model build directory
        model_path_digest = md5sum(model_config["model_file_path"])
        model_output_base_dir = "%s/%s/%s/%s/%s" % (
            FLAGS.output_dir, project_name, "build",
            model_name, model_path_digest)

        if target_abi == "host":
            model_output_dir = "%s/%s" % (model_output_base_dir, target_abi)
        else:
            device_name = sh_commands.adb_get_device_name_by_serialno(serialno)
            model_output_dir = "%s/%s_%s/%s" % (
                model_output_base_dir, device_name.replace(' ', ''),
                target_soc, target_abi)
        model_output_dirs.append(model_output_dir)

        if FLAGS.mode == "build" or FLAGS.mode == "all":
            if os.path.exists(model_output_dir):
                sh.rm("-rf", model_output_dir)
            os.makedirs(model_output_dir)

        model_file_path, weight_file_path = get_model_files(
                model_config["model_file_path"],
                model_output_base_dir,
                model_config["weight_file_path"])

        sh_commands.clear_phone_data_dir(
            target_abi, serialno, phone_data_dir)

        if FLAGS.mode == "build" or FLAGS.mode == "run" or \
                FLAGS.mode == "validate" or \
                FLAGS.mode == "benchmark" or FLAGS.mode == "all":
            sh_commands.gen_random_input(model_output_dir,
                                         model_config["input_nodes"],
                                         model_config["input_shapes"],
                                         input_file_list)

        if FLAGS.mode == "build" or FLAGS.mode == "all":
            build_mace_run_prod(hexagon_mode,
                                model_config["runtime"],
                                target_abi,
                                serialno,
                                vlog_level,
                                embed_model_data,
                                model_output_dir,
                                model_config["input_nodes"],
                                model_config["output_nodes"],
                                model_config["input_shapes"],
                                model_config["output_shapes"],
                                model_name,
                                device_type,
                                FLAGS.round,
                                FLAGS.restart_round,
                                FLAGS.tuning,
                                model_config["limit_opencl_kernel_time"],
                                phone_data_dir,
                                FLAGS.enable_openmp)
            sh_commands.build_benchmark_model(target_abi,
                                              embed_model_data,
                                              model_output_dir,
                                              model_name,
                                              hexagon_mode)

        if FLAGS.mode == "run" or FLAGS.mode == "validate" or \
           FLAGS.mode == "all":
            tuning_run(model_config["runtime"],
                       target_abi,
                       serialno,
                       vlog_level,
                       embed_model_data,
                       model_output_dir,
                       model_config["input_nodes"],
                       model_config["output_nodes"],
                       model_config["input_shapes"],
                       model_config["output_shapes"],
                       model_name,
                       device_type,
                       FLAGS.round,
                       FLAGS.restart_round,
                       FLAGS.out_of_range_check,
                       phone_data_dir,
                       omp_num_threads=FLAGS.omp_num_threads,
                       cpu_affinity_policy=FLAGS.cpu_affinity_policy,
                       gpu_perf_hint=FLAGS.gpu_perf_hint,
                       gpu_priority_hint=FLAGS.gpu_priority_hint)

        if FLAGS.mode == "benchmark":
            gen_opencl_and_tuning_code(
                    target_abi, serialno, [model_output_dir], False)
            sh_commands.benchmark_model(target_abi,
                                        serialno,
                                        vlog_level,
                                        embed_model_data,
                                        model_output_dir,
                                        model_config["input_nodes"],
                                        model_config["output_nodes"],
                                        model_config["input_shapes"],
                                        model_config["output_shapes"],
                                        model_name,
                                        device_type,
                                        phone_data_dir,
                                        FLAGS.omp_num_threads,
                                        FLAGS.cpu_affinity_policy,
                                        FLAGS.gpu_perf_hint,
                                        FLAGS.gpu_priority_hint)

        if FLAGS.mode == "validate" or FLAGS.mode == "all":
            sh_commands.validate_model(target_abi,
                                       serialno,
                                       model_file_path,
                                       weight_file_path,
                                       model_config["platform"],
                                       model_config["runtime"],
                                       model_config["input_nodes"],
                                       model_config["output_nodes"],
                                       model_config["input_shapes"],
                                       model_config["output_shapes"],
                                       model_output_dir,
                                       phone_data_dir,
                                       FLAGS.caffe_env)

    if FLAGS.mode == "build" or FLAGS.mode == "merge" or \
            FLAGS.mode == "all":
        merge_libs_and_tuning_results(
            target_soc,
            target_abi,
            serialno,
            project_name,
            FLAGS.output_dir,
            model_output_dirs,
            hexagon_mode,
            embed_model_data)

    if FLAGS.mode == "throughput_test":
        merged_lib_file = FLAGS.output_dir + \
                "/%s/%s/libmace_%s.%s.a" % \
                (project_name, target_abi, project_name, target_soc)
        first_model = configs["models"].values()[0]
        throughput_test_output_dir = "%s/%s/%s/%s" % (
                FLAGS.output_dir, project_name, "build",
                "throughput_test")
        if os.path.exists(throughput_test_output_dir):
            sh.rm("-rf", throughput_test_output_dir)
        os.makedirs(throughput_test_output_dir)
        input_file_list = model_config["validation_inputs_data"]
        sh_commands.gen_random_input(throughput_test_output_dir,
                                     first_model["input_nodes"],
                                     first_model["input_shapes"],
                                     input_file_list)
        model_tag_dict = {}
        for model_name in configs["models"]:
            runtime = configs["models"][model_name]["runtime"]
            model_tag_dict[runtime] = model_name
        sh_commands.build_run_throughput_test(target_abi,
                                              serialno,
                                              vlog_level,
                                              FLAGS.run_seconds,
                                              merged_lib_file,
                                              throughput_test_output_dir,
                                              embed_model_data,
                                              model_config["input_nodes"],
                                              model_config["output_nodes"],
                                              model_config["input_shapes"],
                                              model_config["output_shapes"],
                                              model_tag_dict.get("cpu", ""),
                                              model_tag_dict.get("gpu", ""),
                                              model_tag_dict.get("dsp", ""),
                                              phone_data_dir)


def main(unused_args):
    common.init_logging()
    configs = parse_model_configs()

    if FLAGS.mode == "validate":
        FLAGS.round = 1
        FLAGS.restart_round = 1

    project_name = os.path.splitext(os.path.basename(FLAGS.config))[0]
    if FLAGS.mode == "build" or FLAGS.mode == "all":
        # Remove previous output dirs
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        elif os.path.exists(os.path.join(FLAGS.output_dir, "libmace")):
            sh.rm("-rf", os.path.join(FLAGS.output_dir, project_name))
            os.makedirs(os.path.join(FLAGS.output_dir, project_name))

        # generate source
        sh_commands.gen_mace_version()
        sh_commands.gen_encrypted_opencl_source()
        sh_commands.gen_mace_engine_creator_source(configs['models'].keys())

    embed_model_data = configs["embed_model_data"]
    target_socs = get_target_socs(configs)

    vlog_level = FLAGS.vlog_level
    phone_data_dir = "/data/local/tmp/mace_run/"

    if FLAGS.mode == "build" or FLAGS.mode == "all":
        print '* Model Convert'
        sh_commands.clear_model_codegen()
        for model_name in configs["models"]:
            print '===================', model_name, '==================='
            model_config = configs["models"][model_name]
            data_type, device_type = get_data_and_device_type(
                model_config["runtime"])

            # Create model build directory
            model_path_digest = md5sum(model_config["model_file_path"])

            model_output_base_dir = "%s/%s/%s/%s/%s" % (
                FLAGS.output_dir, project_name, "build",
                model_name, model_path_digest)

            if os.path.exists(model_output_base_dir):
                sh.rm("-rf", model_output_base_dir)
            os.makedirs(model_output_base_dir)

            model_file_path, weight_file_path = get_model_files(
                model_config["model_file_path"],
                model_output_base_dir,
                model_config["weight_file_path"])

            sh_commands.gen_model_code(
                "mace/codegen/models/%s" % model_name,
                model_config["platform"],
                model_file_path,
                weight_file_path,
                model_config["model_sha256_checksum"],
                ",".join(model_config["input_nodes"]),
                ",".join(model_config["output_nodes"]),
                data_type,
                model_config["runtime"],
                model_name,
                ":".join(model_config["input_shapes"]),
                model_config["dsp_mode"],
                embed_model_data,
                model_config["fast_conv"],
                model_config["obfuscate"])

    for target_abi in configs["target_abis"]:
        for target_soc in target_socs:
            if target_abi != 'host':
                serialnos = sh_commands.get_target_socs_serialnos([target_soc])
                for serialno in serialnos:
                    props = sh_commands.adb_getprop_by_serialno(serialno)
                    print(
                        "===================================================="
                    )
                    print("Trying to lock device %s" % serialno)
                    with sh_commands.device_lock(serialno):
                        print("Run on device: %s, %s, %s" % (
                            serialno, props["ro.board.platform"],
                              props["ro.product.model"]))
                        process_models(project_name, configs, embed_model_data,
                                       vlog_level, target_abi, phone_data_dir,
                                       target_soc, serialno)
            else:
                print("====================================================")
                print("Run on host")
                process_models(project_name, configs, embed_model_data,
                               vlog_level, target_abi, phone_data_dir)

    if FLAGS.mode == "build" or FLAGS.mode == "all":
        sh_commands.packaging_lib(FLAGS.output_dir, project_name)


if __name__ == "__main__":
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)
