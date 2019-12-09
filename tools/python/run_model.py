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

from py_proto import mace_pb2
from utils import util
from utils import device
from utils import config_parser
from utils.config_parser import DeviceType
from utils.target import Target
from utils.config_parser import ModelKeys
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
        run_models_for_device(flags, args, dev)


def run_models_for_device(flags, args, dev):
    conf = config_parser.parse(flags.config)
    for name, model_conf in conf["models"].items():
        if not flags.model_name or name == flags.model_name:
            MaceLogger.info("Run model %s" % name)
            model_conf = config_parser.normalize_model_config(model_conf)
            run_model_for_device(flags, args, dev, name, model_conf)


def run_model_for_device(flags, args, dev, model_name, model_conf):
    runtime = flags.runtime
    target_abi = flags.target_abi
    install_dir = run_target.default_install_dir(target_abi) + "/" + model_name
    sysdir = install_dir + "/interior"
    dev.mkdir(sysdir)

    if not runtime:
        runtime = model_conf[ModelKeys.runtime]
        if runtime == DeviceType.CPU_GPU:
            runtime = DeviceType.GPU
    else:
        runtime = config_parser.parse_device_type(runtime)

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
    model_args = {"model_name": model_name,
                  "model_file": model_file_path,
                  "model_data_file": model_data_file_path,
                  "input_node": ",".join(
                      model_conf[ModelKeys.input_tensors]),
                  "input_shape": join_2d_array(
                      model_conf[ModelKeys.input_shapes]),
                  "output_node": ",".join(
                      model_conf[ModelKeys.output_tensors]),
                  "output_shape": join_2d_array(
                      model_conf[ModelKeys.output_shapes]),
                  "input_data_format": ",".join(
                      [df.name for df in
                       model_conf[ModelKeys.input_data_formats]]),
                  "output_data_format": ",".join(
                      [df.name for df in
                       model_conf[ModelKeys.output_data_formats]]),
                  "device": runtime.name
                  }

    opts = ["--%s=%s" % (arg_key, arg_val) for arg_key, arg_val in
            model_args.items()] + args
    should_generate_data = (flags.validate
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
                                model_conf[ModelKeys.input_shapes],
                                model_conf[ModelKeys.input_ranges],
                                model_conf[ModelKeys.input_data_types])

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
    if flags.vlog_level > 0:
        envs += ["MACE_CPP_MIN_VLOG_LEVEL=%s" % flags.vlog_level]

    build_dir = flags.build_dir + "/" + target_abi
    libs = []
    if model_conf[ModelKeys.runtime] == DeviceType.HEXAGON:
        libs += ["third_party/nnlib/%s/libhexagon_controller.so" % target_abi]
    elif model_conf[ModelKeys.runtime] == DeviceType.APU:
        libs += ["third_party/apu/libapu-frontend.so"]

    target = Target(build_dir + "/install/bin/mace_run", libs,
                    opts=opts, envs=envs)
    run_target.run_target(target_abi, install_dir, target,
                          device_ids=flags.target_socs)

    if runtime == DeviceType.GPU:
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
        validate.validate(model_conf[ModelKeys.platform],
                          validate_model_file,
                          validate_weight_file,
                          input_file_prefix,
                          output_file_prefix,
                          model_conf[ModelKeys.input_shapes],
                          model_conf[ModelKeys.output_shapes],
                          model_conf[ModelKeys.input_data_formats],
                          model_conf[ModelKeys.output_data_formats],
                          model_conf[ModelKeys.input_tensors],
                          model_conf[ModelKeys.output_tensors],
                          flags.validate_threshold,
                          model_conf[ModelKeys.input_data_types],
                          flags.backend,
                          "",
                          "")
    if should_generate_data:
        shutil.rmtree(tmpdirname)


def generate_input_data(input_file, input_node, input_shape, input_ranges,
                        input_data_type):
    np.random.seed()
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
        "--model_name",
        type=str,
        default="",
        help="model name in yaml conf"
    )
    parser.add_argument(
        "--target_abi",
        type=str,
        default="armeabi-v7a",
        help="Target ABI: host, armeabi-v7a, arm64-v8a,"
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
        default="tensorflow",
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

    return parser.parse_known_args()


if __name__ == "__main__":
    flags, args = parse_args()
    if flags.build:
        build_engine(flags)
    run_models(flags, args)
