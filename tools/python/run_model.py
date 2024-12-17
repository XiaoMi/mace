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
import os
import tempfile
import shutil
import numpy as np

from apu_utils import get_apu_so_paths
from py_proto import mace_pb2
from utils import util
from utils import device
from utils import config_parser
from utils.config_parser import DeviceType
from utils.target import Target
from utils.config_parser import ModelKeys
from utils.util import ABIType
from utils.util import MaceLogger
from utils.util import mace_check
import run_target
import validate


"""
Tool for mace_run:

python tools/python/run_model.py \
--config ../mace-models/mobilenet-v1/mobilenet-v1.yml --build --validate
python tools/python/run_model.py \
--config ../mace-models/mobilenet-v1/mobilenet-v1.yml --benchmark
python tools/python/run_model.py \
--config ../mace-models/mobilenet-v1/mobilenet-v1.yml --runtime=cpu

"""


def join_2d_array(xs):
    return ":".join([",".join([str(y) for y in x]) for x in xs])


def build_engine(flags):
    cmake_shell = os.path.abspath(
        os.path.dirname(
            __file__)) + "/../cmake/cmake-build-%s.sh" % flags.target_abi
    os.environ["BUILD_DIR"] = flags.build_dir + "/" + flags.target_abi
    if flags.runtime:
        os.environ["RUNTIME"] = config_parser.parse_device_type(
            flags.runtime).name
    if flags.gencode_model:
        os.environ["RUNMODE"] = "code"
    device.execute("bash " + cmake_shell)


def run_models(flags, args):
    if flags.device_conf:
        device_conf = config_parser.parse_device_info(flags.device_conf)
        device.ArmLinuxDevice.set_devices(device_conf)

    run_devices = device.choose_devices(flags.target_abi, flags.target_socs)
    MaceLogger.info("Run on devices: %s" % run_devices)

    for device_id in run_devices:
        dev = device.create_device(flags.target_abi, device_id)
        if len(flags.devices_to_run) > 0 and \
                device_id not in flags.devices_to_run.split(','):
            continue
        run_models_for_device(flags, args, dev)


def run_models_for_device(flags, args, dev):
    conf = config_parser.parse(flags.config)
    for name, model_conf in conf["models"].items():
        if not flags.model_name or name == flags.model_name:
            MaceLogger.info("Run model %s" % name)
            model_conf = config_parser.normalize_model_config(model_conf)
            run_model_for_device(flags, args, dev, name, model_conf)


def run_model_for_device(flags, args, dev, model_name, model_conf):
    target_abi = flags.target_abi
    install_dir = run_target.default_install_dir(target_abi) + "/" + model_name
    sysdir = install_dir + "/interior"
    dev.mkdir(sysdir)

    runtime_list = []
    for graph_name, graph_conf in model_conf[ModelKeys.subgraphs].items():
        runtime = graph_conf[ModelKeys.runtime]
        runtime_list.append(runtime)
        mace_check(runtime != DeviceType.APU or target_abi == "arm64-v8a",
                   "APU runtime does only support arm64-v8a")

    # install models to devices
    workdir = flags.output + "/" + model_name
    model_file = model_name + ".pb"
    model_data_file = model_name + ".data"
    model_path = workdir + "/model/" + model_file
    model_data_path = workdir + "/model/" + model_data_file
    if os.path.exists(model_path) and os.path.exists(model_data_path):
        dev.install(Target(model_path), install_dir)
        dev.install(Target(model_data_path), install_dir)
    else:
        MaceLogger.warning("No models exist in %s, use --model_file and"
                           " --model_data_file specified in args" % model_path)

    if ModelKeys.check_tensors in model_conf:
        model_conf[ModelKeys.output_tensors] = model_conf[
            ModelKeys.check_tensors]
        model_conf[ModelKeys.output_shapes] = model_conf[
            ModelKeys.check_shapes]

    model_file_path = ""
    if not flags.gencode_model:
        model_file_path = install_dir + "/" + model_file
    model_data_file_path = ""
    if not flags.gencode_param:
        model_data_file_path = install_dir + "/" + model_data_file

    input_tensors_info = config_parser.find_input_tensors_info(
        model_conf[ModelKeys.subgraphs], model_conf[ModelKeys.input_tensors])
    output_tensors_info = config_parser.find_output_tensors_info(
        model_conf[ModelKeys.subgraphs], model_conf[ModelKeys.output_tensors])

    if ModelKeys.check_tensors in model_conf[ModelKeys.subgraphs][ModelKeys.default_graph].keys():  
        output_tensors_info[ModelKeys.output_shapes] = model_conf[ModelKeys.subgraphs][ModelKeys.default_graph][ModelKeys.check_shapes]
        output_tensors_info[ModelKeys.output_tensors] = model_conf[ModelKeys.subgraphs][ModelKeys.default_graph][ModelKeys.check_tensors]
        model_conf[ModelKeys.output_tensors] = output_tensors_info[ModelKeys.output_tensors]

    model_args = {"model_name": model_name,
                  "model_file": model_file_path,
                  "model_data_file": model_data_file_path,
                  "input_node": ",".join(
                      model_conf[ModelKeys.input_tensors]),
                  "input_shape": join_2d_array(
                      input_tensors_info[ModelKeys.input_shapes]),
                  "output_node": ",".join(
                      model_conf[ModelKeys.output_tensors]),
                  "output_shape": join_2d_array(
                      output_tensors_info[ModelKeys.output_shapes]),
                  "input_data_format": ",".join(
                      [df.name for df in
                       input_tensors_info[ModelKeys.input_data_formats]]),
                  "output_data_format": ",".join(
                      [df.name for df in
                       output_tensors_info[ModelKeys.output_data_formats]]),
                  "input_data_type": ",".join(
                      [str(i) for i in
                       input_tensors_info[ModelKeys.input_data_types]]),
                  "output_data_type": ",".join(
                      [str(i) for i in
                       output_tensors_info[ModelKeys.output_data_types]]
                  ) 
                }

    opts = ["--%s='%s'" % (arg_key, arg_val) for arg_key, arg_val in
            model_args.items()] + args
    should_generate_data = (flags.validate or flags.quantize_stat
                            or flags.tune or "--benchmark" in opts)

    if should_generate_data:
        tmpdirname = tempfile.mkdtemp()
        input_file_prefix = tmpdirname + "/" + model_name

        if ModelKeys.validation_inputs_data in model_conf:
            input_tensor = model_conf[ModelKeys.input_tensors]
            input_data = model_conf[ModelKeys.validation_inputs_data]
            mace_check(len(input_tensor) == len(input_data),
                       "len(input_tensor) != len(validate_data")

            for i in range(len(input_tensor)):
                util.download_or_get_file(
                    model_conf[ModelKeys.validation_inputs_data][i], "",
                    util.formatted_file_name(input_file_prefix,
                                             input_tensor[i]))
        else:
            generate_input_data(input_file_prefix,
                                model_conf[ModelKeys.input_tensors],
                                input_tensors_info[ModelKeys.input_shapes],
                                input_tensors_info[ModelKeys.input_ranges],
                                input_tensors_info[ModelKeys.input_data_types])

        dev.install(Target(tmpdirname), install_dir + "/validate_in")
        target_input_file = "%s/validate_in/%s" % (
            install_dir, model_name)
        target_output_dir = "%s/validate_out" % install_dir
        dev.mkdir(target_output_dir)
        target_output_file = target_output_dir + "/" + model_name
        opts += ["--input_file=%s" % target_input_file,
                 "--output_file=%s" % target_output_file]

    # run
    envs = flags.envs.split(" ") + ["MACE_INTERNAL_STORAGE_PATH=%s" % sysdir]
    if flags.tune:
        envs += ["MACE_TUNING=1",
                 "MACE_RUN_PARAMETER_PATH=%s/interior/tune_params"
                 % install_dir]
        opts += ["--round=0"]

    mace_check(flags.vlog_level >= 0,
               "vlog_level should be greater than zeror")
    envs += ["MACE_CPP_MIN_VLOG_LEVEL=%s" % flags.vlog_level]
    if flags.quantize_stat:
        envs += ["MACE_LOG_TENSOR_RANGE=1"]

    build_dir = flags.build_dir + "/" + target_abi
    libs = []
    if DeviceType.HEXAGON in runtime_list:
        libs += ["third_party/nnlib/%s/libhexagon_controller.so" % target_abi]
        if target_abi == "arm64-v8a":
            libs += ["third_party/nnlib/v66/libhexagon_nn_skel.so"]
    elif runtime == DeviceType.HTA:
        libs += ["third_party/hta/%s/libhta_hexagon_runtime.so" % target_abi]
    elif DeviceType.APU in runtime_list:
        apu_libs = get_apu_so_paths(dev)
        libs += apu_libs
    elif DeviceType.HTP in runtime_list:
        libs += [
            "third_party/qnn/target/%s/libQnnHtpAltPrepStub.so" %
            target_abi]
        libs += [
            "third_party/qnn/target/%s/libQnnHtpAltPrepV69Stub.so" %
            target_abi]
    cpp_shared_lib_path = os.path.join(
        build_dir, "install/lib/libc++_shared.so")
    if os.path.exists(cpp_shared_lib_path):
        libs.append(cpp_shared_lib_path)

    target = Target(build_dir + "/install/bin/mace_run", libs,
                    opts=opts, envs=envs)
    run_target.run_target(target_abi, install_dir, target, dev)

    if DeviceType.GPU in runtime_list:
        opencl_dir = workdir + "/opencl"
        util.mkdir_p(opencl_dir)
        dev.pull(
            Target(install_dir + "/interior/mace_cl_compiled_program.bin"),
            "%s/%s_compiled_opencl_kernel.%s.%s.bin" % (
                opencl_dir, model_name,
                dev.info()["ro.product.model"].replace(' ', ''),
                dev.info()["ro.board.platform"]))
        if flags.tune:
            dev.pull(Target(install_dir + "/interior/tune_params"),
                     "%s/%s_tuned_opencl_parameter.%s.%s.bin" % (
                         opencl_dir, model_name,
                         dev.info()["ro.product.model"].replace(' ', ''),
                         dev.info()["ro.board.platform"]))
    
    if flags.validate:
        validate_model_file = util.download_or_get_model(
            model_conf[ModelKeys.model_file_path],
            model_conf[ModelKeys.model_sha256_checksum],
            tmpdirname)
        validate_weight_file = ""
        if ModelKeys.weight_file_path in model_conf:
            validate_weight_file = util.download_or_get_model(
                model_conf[ModelKeys.weight_file_path],
                model_conf[ModelKeys.weight_sha256_checksum],
                tmpdirname)
        dev.pull(Target(target_output_dir), tmpdirname + "/validate_out")
        output_file_prefix = tmpdirname + "/validate_out/" + model_name
        validation_outputs_data = \
            model_conf.get(ModelKeys.validation_outputs_data, [])
        validate.validate(model_conf[ModelKeys.platform],
                          validate_model_file,
                          validate_weight_file,
                          input_file_prefix,
                          output_file_prefix,
                          input_tensors_info[ModelKeys.input_shapes],
                          output_tensors_info[ModelKeys.output_shapes],
                          input_tensors_info[ModelKeys.input_data_formats],
                          output_tensors_info[ModelKeys.output_data_formats],
                          input_tensors_info[ModelKeys.input_tensors],
                          output_tensors_info[ModelKeys.output_tensors],
                          flags.validate_threshold,
                          input_tensors_info[ModelKeys.input_data_types],
                          flags.backend,
                          validation_outputs_data,
                          flags.accuracy_log)
    if should_generate_data:
        shutil.rmtree(tmpdirname)


def generate_input_data(input_file, input_node, input_shape, input_ranges,
                        input_data_type):
    np.random.seed()
    print("The scope of generated data: ", input_ranges)
    for i in range(len(input_node)):
        data = np.random.random(input_shape[i]) * (
            input_ranges[i][1] - input_ranges[i][0]) + input_ranges[i][0]
        input_file_name = util.formatted_file_name(input_file, input_node[i])
        MaceLogger.info('Generate input file: %s' % input_file_name)
        if input_data_type[i] == mace_pb2.DT_FLOAT:
            np_data_type = np.float32
        elif input_data_type[i] == mace_pb2.DT_INT32:
            np_data_type = np.int32

        data.astype(np_data_type).tofile(input_file_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="yaml conf path"
    )
    parser.add_argument(
        '--justprint',
        action="store_true",
        help="just print the commands but do nothing(for debug)")
    parser.add_argument(
        '--norun',
        action="store_true",
        help="do all the thing except running the model")
    parser.add_argument(
        '--devices_to_run',
        type=str,
        default="",
        help="devices you want to run")
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="model name in yaml conf"
    )
    parser.add_argument(
        "--target_abi",
        type=str,
        default="arm64-v8a",
        help="Target ABI: host, qnx, armeabi-v7a, arm64-v8a,"
             " arm-linux-gnueabihf, aarch64-linux-gnu"
    )
    parser.add_argument(
        "--target_socs",
        type=str,
        default="all",
        help="serialno for adb connection,"
             " username@ip for arm linux,"
             " host for host"
             " | all | random"
    )
    parser.add_argument(
        "--device_conf",
        type=str,
        default="",
        help="device yaml config path"
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default="",
        help="cpu/gpu/dsp/hta/apu"
    )
    parser.add_argument("--envs", type=str, default="",
                        help="Environment vars: "
                             " MACE_OUT_OF_RANGE_CHECK=1, "
                             " MACE_OPENCL_PROFILING=1,"
                             " MACE_INTERNAL_STORAGE_PATH=/path/to,"
                             " LD_PRELOAD=/path/to")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="enable validate"
    )
    parser.add_argument(
        "--validate_threshold",
        type=float,
        default="0.99",
        help="validate threshold"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        help="onnx backend framework")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="enable tuning"
    )
    parser.add_argument(
        "--build_dir",
        type=str,
        default="build/cmake-build",
        help="cmake build dir"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="if build before run"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="build",
        help="output dir")
    parser.add_argument(
        '--vlog_level',
        type=int,
        default="0",
        help="vlog level")
    parser.add_argument(
        "--gencode_model",
        action="store_true",
        help="use compiled model")
    parser.add_argument(
        "--gencode_param",
        action="store_true",
        help="use compiled param")
    parser.add_argument(
        "--quantize_stat",
        action="store_true",
        help="whether to stat quantization range.")
    parser.add_argument(
        "--accuracy_log",
        type=str,
        default="",
        help="the file of result of validate accuracy")

    return parser.parse_known_args()


if __name__ == "__main__":
    flags, args = parse_args()
    if flags.build:
        build_engine(flags)
    device.justprint = flags.justprint
    device.norun = flags.norun
    run_models(flags, args)
