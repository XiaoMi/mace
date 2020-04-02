# Copyright 2020 The MACE Authors. All Rights Reserved.
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
import copy
import numpy as np
import shutil
import tempfile

from micro_converter import MicroConverter
from py_proto import mace_pb2
import run_target
from utils import util
from utils import device
from utils import config_parser
from utils.target import Target
from utils.config_parser import ModelKeys
from utils.util import MaceLogger
from utils.util import mace_check
import validate
import layers_validate


def join_2d_array(xs):
    return ":".join([",".join([str(y) for y in x]) for x in xs])


def build_engine(model_name, data_type):
    mace_check(flags.model_name is not None and len(model_name) > 0,
               "you should specify model name for build.")
    command = "bazel build //micro/tools:micro_run_static" \
              " --config optimization " \
              " --copt \"-DMICRO_MODEL_NAME=%s\"" % model_name
    if data_type == mace_pb2.DT_BFLOAT16:
        command += " --copt \"-DMACE_ENABLE_BFLOAT16\""
        print("The current engine's data type is bfloat16.")
    device.execute(command)


def get_model_conf_by_name(flags, conf):
    for name, model_conf in conf["models"].items():
        if not flags.model_name or name == flags.model_name:
            return model_conf
    return None


def run_model(flags, args, conf):
    model_conf = get_model_conf_by_name(flags, conf)
    mace_check(model_conf is not None, "Get model conf failed.")
    model_conf = config_parser.normalize_model_config(model_conf)
    run_model_with_conf(flags, args, flags.model_name, model_conf)


def gen_sub_model_conf(output_config, flags, conf):
    model_conf = copy.deepcopy(get_model_conf_by_name(flags, conf))
    model_conf['subgraphs'][0]['output_tensors'] = \
        output_config['output_tensors']
    model_conf['subgraphs'][0]['output_shapes'] = \
        output_config['output_shapes']
    return model_conf


def run_layers_validate(flags, args, original_conf):
    model_name = flags.model_name
    original_model_dir = flags.output + "/" + \
        original_conf['library_name'] + "/model"
    model_dir = "/tmp/micro_run/model"
    device.execute("mkdir -p %s" % model_dir)
    device.execute("cp -p %s/%s.pb %s" %
                   (original_model_dir, model_name, model_dir))
    params_file_path = "%s/%s.data" % (original_model_dir, model_name)
    output_configs = layers_validate.get_layers(
        model_dir, model_name, flags.layers)

    for i in range(len(output_configs)):
        sub_model_conf = gen_sub_model_conf(
            output_configs[i], flags, original_conf)
        with open(output_configs[i]['model_file_path'], "rb") as model_file:
            net_def = mace_pb2.NetDef()
            net_def.ParseFromString(model_file.read())
            with open(params_file_path, "rb") as params_file:
                weights = bytearray(params_file.read())
                micro_conf = \
                    config_parser.normalize_model_config(sub_model_conf)
                MicroConverter(micro_conf, net_def,
                               weights, model_name).gen_code()
                build_engine(model_name, micro_conf[ModelKeys.data_type])
                run_model_with_conf(flags, args, model_name, micro_conf)


def run_model_with_conf(flags, args, model_name, model_conf):
    target_abi = "host"
    dev = device.HostDevice("host", target_abi)
    install_dir = "/tmp/micro_run/" + model_name

    if ModelKeys.check_tensors in model_conf:
        model_conf[ModelKeys.output_tensors] = model_conf[
            ModelKeys.check_tensors]
        model_conf[ModelKeys.output_shapes] = model_conf[
            ModelKeys.check_shapes]

    model_args = {"model_name": model_name,
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
                       model_conf[ModelKeys.output_data_formats]])
                  }

    opts = ["--%s=%s" % (arg_key, arg_val) for arg_key, arg_val in
            model_args.items()] + args

    # generate data start
    tmp_dir_name = tempfile.mkdtemp()
    input_file_prefix = tmp_dir_name + "/" + model_name
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

    dev.install(Target(tmp_dir_name), install_dir + "/validate_in")
    target_input_file = "%s/validate_in/%s" % (
        install_dir, model_name)
    target_output_dir = "%s/validate_out" % install_dir
    dev.mkdir(target_output_dir)
    target_output_file = target_output_dir + "/" + model_name
    opts += ["--input_file=%s" % target_input_file,
             "--output_file=%s" % target_output_file]
    # generate data end

    envs = []
    if flags.vlog_level > 0:
        envs += ["MACE_CPP_MIN_VLOG_LEVEL=%s" % flags.vlog_level]

    target = Target("bazel-bin/micro/tools/micro_run_static", [],
                    opts=opts, envs=envs)
    run_target.run_target(target_abi, install_dir, target,
                          device_ids="host")

    if flags.validate:
        validate_model_file = util.download_or_get_model(
            model_conf[ModelKeys.model_file_path],
            model_conf[ModelKeys.model_sha256_checksum],
            tmp_dir_name)

        validate_weight_file = ""
        if ModelKeys.weight_file_path in model_conf:
            validate_weight_file = util.download_or_get_model(
                model_conf[ModelKeys.weight_file_path],
                model_conf[ModelKeys.weight_sha256_checksum],
                tmp_dir_name)

        dev.pull(Target(target_output_dir), tmp_dir_name + "/validate_out")
        output_file_prefix = tmp_dir_name + "/validate_out/" + model_name
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
    shutil.rmtree(tmp_dir_name)


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
        "--layers",
        type=str,
        default="-1",
        help="'start_layer:end_layer' or 'layer', similar to python slice."
             " Use with --validate flag.")
    parser.add_argument(
        "--backend",
        type=str,
        default="tensorflow",
        help="onnx backend framework")
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

    return parser.parse_known_args()


if __name__ == "__main__":
    flags, args = parse_args()
    conf = config_parser.parse(flags.config)
    if flags.build or flags.validate:
        micro_conf = config_parser.normalize_model_config(
            conf[ModelKeys.models][flags.model_name])
        build_engine(flags.model_name, micro_conf[ModelKeys.data_type])
    if flags.validate and flags.layers != "-1":
        run_layers_validate(flags, args, conf)
    else:
        run_model(flags, args, conf)
