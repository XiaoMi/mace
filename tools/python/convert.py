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

# python tools/python/convert.py \
# --config ../mace-models/mobilenet-v2/mobilenet-v2.yml

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import sys

from micro_converter import MicroConverter
from py_proto import mace_pb2
from utils import config_parser
from utils.config_parser import DataFormat
from utils.config_parser import DeviceType
from utils.config_parser import Platform
from utils import util
from utils.util import mace_check
from utils.config_parser import normalize_model_config
from utils.config_parser import ModelKeys
from utils.convert_util import merge_params
from transform import base_converter as cvt
from transform import transformer
from visualize import visualize_model


def transpose_shape(shape, dst_order):
    t_shape = [0] * len(shape)
    for i in range(len(shape)):
        t_shape[i] = shape[dst_order[i]]
    return t_shape


def convert_micro(model_name, net_confs, net_def,
                  params, model_output):
    mace_check(len(net_confs.items()) == 1,
               "Multi graph not supported in Micro")
    for graph_name, graph_conf in net_confs.items():
        micro_converter = MicroConverter(graph_conf, copy.deepcopy(net_def),
                                         copy.deepcopy(params), model_name)
        micro_converter.gen_code()
        micro_converter.package(model_output + "/" +
                                model_name + "_micro.tar.gz")


def add_input_output_tensor(multi_net_def, model_conf):
    if len(model_conf[ModelKeys.subgraphs]) == 1:
        model_conf = model_conf[ModelKeys.subgraphs][ModelKeys.default_graph]
    for input_tensor in model_conf[ModelKeys.input_tensors]:
        multi_net_def.input_tensor.append(input_tensor)
    for output_tensor in model_conf[ModelKeys.output_tensors]:
        multi_net_def.output_tensor.append(output_tensor)


def convert(conf, output, enable_micro=False):
    for model_name, model_conf in conf["models"].items():
        model_output = output + "/" + model_name + "/model"
        org_model_dir = output + "/" + model_name + "/org_model"
        util.mkdir_p(model_output)
        util.mkdir_p(org_model_dir)

        model_conf = normalize_model_config(model_conf, model_output,
                                            org_model_dir)
        conf["models"][model_name] = model_conf
        net_confs = model_conf[ModelKeys.subgraphs]

        model = mace_pb2.MultiNetDef()
        add_input_output_tensor(model, model_conf)

        model_params = []
        for net_name, net_conf in net_confs.items():
            if "quantize_stat" in conf:
                net_conf["quantize_stat"] = conf["quantize_stat"]
            net_def_with_Data = convert_net(net_name, net_conf, enable_micro)
            try:
                visualizer = visualize_model.ModelVisualizer(
                    net_name, net_def_with_Data, model_output)
                visualizer.save_html()
            except:  # noqa
                print("Failed to visualize graph:", sys.exc_info())
            net_def, params = merge_params(net_def_with_Data,
                                           net_conf[ModelKeys.data_type])
            if enable_micro:
                convert_micro(model_name, net_confs, net_def,
                              params, model_output,)

            net_def.data_offset = len(model_params)
            net_def.data_size = len(params)
            model.net_def.extend([net_def])
            model_params.extend(params)
        # store model and weight to files
        output_model_file = model_output + "/" + model_name + ".pb"
        output_params_file = model_output + "/" + model_name + ".data"
        with open(output_model_file, "wb") as f:
            f.write(model.SerializeToString())
        with open(output_params_file, "wb") as f:
            f.write(bytearray(model_params))
        with open(output_model_file + "_txt", "w") as f:
            f.write(str(model))


def convert_net(net_name, conf, enable_micro):
    option = cvt.ConverterOption()
    option.name = net_name
    option.order = conf.get(ModelKeys.order, 0)
    if ModelKeys.quantize_stat in conf:
        option.quantize_stat = conf[ModelKeys.quantize_stat]
    else:
        option.quantize_stat = False

    if ModelKeys.graph_optimize_options in conf:
        option.transformer_option = conf[ModelKeys.graph_optimize_options]
    if ModelKeys.winograd in conf:
        option.winograd = conf[ModelKeys.winograd]
    if ModelKeys.quantize in conf:
        option.quantize = conf[ModelKeys.quantize]
    if ModelKeys.quantize_schema in conf:
        option.quantize_schema = conf[ModelKeys.quantize_schema]
    if ModelKeys.quantize_large_weights in conf:
        option.quantize_large_weights = conf[ModelKeys.quantize_large_weights]
    if ModelKeys.quantize_range_file in conf:
        option.quantize_range_file = conf[ModelKeys.quantize_range_file]
    if ModelKeys.change_concat_ranges in conf:
        option.change_concat_ranges = conf[ModelKeys.change_concat_ranges]
    if ModelKeys.cl_mem_type in conf:
        option.cl_mem_type = conf[ModelKeys.cl_mem_type]
    if ModelKeys.platform in conf:
        option.platform = conf[ModelKeys.platform]
    if ModelKeys.runtime in conf:
        option.device = conf[ModelKeys.runtime]
        if option.device == DeviceType.CPU_GPU:
            # when convert, cpu and gpu share the same model
            option.device = DeviceType.CPU
        # we don't need `value`, but to be consistent with legacy code
        # used by `base_converter`
        option.device = option.device.value

    option.enable_micro = enable_micro
    option.data_type = conf[ModelKeys.data_type]

    for i in range(len(conf[ModelKeys.input_tensors])):
        input_node = cvt.NodeInfo()
        input_node.name = conf[ModelKeys.input_tensors][i]
        if ModelKeys.input_aliases in conf:
            input_node.alias = conf[ModelKeys.input_aliases][i]
        else:
            input_node.alias = input_node.name
        input_node.shape = conf[ModelKeys.input_shapes][i]
        input_node.data_type = conf[ModelKeys.input_data_types][i]
        input_node.data_format = conf[ModelKeys.input_data_formats][i]
        if (input_node.data_format == DataFormat.NCHW and len(
                input_node.shape) == 4):
            input_node.shape = transpose_shape(input_node.shape, [0, 2, 3, 1])
            input_node.data_format = DataFormat.NHWC
        input_node.range = conf[ModelKeys.input_ranges][i]
        option.add_input_node(input_node)

    for i in range(len(conf[ModelKeys.output_tensors])):
        output_node = cvt.NodeInfo()
        output_node.name = conf[ModelKeys.output_tensors][i]
        if ModelKeys.output_aliases in conf:
            output_node.alias = conf[ModelKeys.output_aliases][i]
        else:
            output_node.alias = output_node.name
        output_node.shape = conf[ModelKeys.output_shapes][i]
        output_node.data_type = conf[ModelKeys.output_data_types][i]
        output_node.data_format = conf[ModelKeys.output_data_formats][i]
        if output_node.data_format == DataFormat.NCHW and len(
                output_node.shape) == 4:
            output_node.shape = transpose_shape(output_node.shape,
                                                [0, 2, 3, 1])
            output_node.data_format = DataFormat.NHWC
        option.add_output_node(output_node)

    if ModelKeys.check_tensors in conf:
        for i in range(len(conf[ModelKeys.check_tensors])):
            check_node = cvt.NodeInfo()
            check_node.name = conf[ModelKeys.check_tensors][i]
            check_node.shape = conf[ModelKeys.check_shapes][i]
            option.add_check_node(check_node)
    else:
        option.check_nodes = option.output_nodes

    option.build()

    print("Transform model to one that can better run on device")
    platform = conf[ModelKeys.platform]
    if platform == Platform.TENSORFLOW:
        from transform import tensorflow_converter
        converter = tensorflow_converter.TensorflowConverter(
            option, conf["model_file_path"])
    elif platform == Platform.CAFFE:
        from transform import caffe_converter
        converter = caffe_converter.CaffeConverter(option,
                                                   conf["model_file_path"],
                                                   conf["weight_file_path"])
    elif platform == Platform.ONNX:
        from transform import onnx_converter
        converter = onnx_converter.OnnxConverter(option,
                                                 conf["model_file_path"])
    elif platform == Platform.MEGENGINE:
        from transform import megengine_converter
        converter = megengine_converter.MegengineConverter(
            option, conf["model_file_path"])
    elif platform == Platform.KERAS:
        from transform import keras_converter
        converter = keras_converter.KerasConverter(
            option, conf["model_file_path"])
    elif platform == Platform.PYTORCH:
        from transform import pytorch_converter
        converter = pytorch_converter.PytorchConverter(
            option, conf["model_file_path"])
    else:
        mace_check(False, "Mace do not support platorm %s yet." % platform)

    # Converter needs to pass some info to transformer by converter_info.
    output_graph_def, converter_info = converter.run()
    mace_transformer = transformer.Transformer(option, output_graph_def,
                                               converter_info)
    output_graph_def, quantize_activation_info = mace_transformer.run()

    runtime = conf[ModelKeys.runtime]
    if runtime in [DeviceType.HEXAGON,
                   DeviceType.HTA]:
        from transform import hexagon_converter
        converter = hexagon_converter.HexagonConverter(
            option, output_graph_def, quantize_activation_info)
        output_graph_def = converter.run()
    elif runtime == DeviceType.APU:
        mace_check(platform == Platform.TENSORFLOW,
                   "apu only support model from tensorflow")
        from transform import apu_converter
        converter = apu_converter.ApuConverter(
            option, output_graph_def, quantize_activation_info)
        output_graph_def = converter.run()

    return output_graph_def


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default="",
        required=True,
        help="the path of model yaml configuration file.")
    parser.add_argument(
        '--output',
        type=str,
        default="build",
        help="output dir")
    parser.add_argument(
        "--enable_micro",
        action="store_true",
        help="enable convert micro.")
    flgs, _ = parser.parse_known_args()
    return flgs


if __name__ == '__main__':
    flags = parse_args()
    conf = config_parser.parse(flags.config)
    convert(conf, flags.output, flags.enable_micro)
