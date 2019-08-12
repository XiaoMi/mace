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
import sys
import numpy as np
from utils import config_parser
from utils import util
from utils.util import mace_check
from py_proto import mace_pb2
from transform import base_converter as cvt
from transform import transformer
from visualize import visualize_model

device_type_map = {'cpu': cvt.DeviceType.CPU.value,
                   'gpu': cvt.DeviceType.GPU.value,
                   'dsp': cvt.DeviceType.HEXAGON.value,
                   'hta': cvt.DeviceType.HTA.value,
                   'apu': cvt.DeviceType.APU.value,
                   'cpu+gpu': cvt.DeviceType.CPU.value}

data_format_map = {
    'NONE': cvt.DataFormat.NONE,
    'NHWC': cvt.DataFormat.NHWC,
    'NCHW': cvt.DataFormat.NCHW,
    'OIHW': cvt.DataFormat.OIHW,
}

data_type_map = {
    'float32': mace_pb2.DT_FLOAT,
    'int32': mace_pb2.DT_INT32,
}


def parse_data_type(data_type, quantize):
    if quantize or data_type == 'fp32_fp32':
        return mace_pb2.DT_FLOAT
    else:
        return mace_pb2.DT_HALF


def split_shape(shape):
    if shape.strip() == "":
        return []
    else:
        return shape.split(',')


def parse_int_array_from_str(ints_str):
    return [int(i) for i in split_shape(ints_str)]


def parse_float_array_from_str(floats_str):
    return [float(i) for i in floats_str.split(',')]


def transpose_shape(shape, dst_order):
    t_shape = [0] * len(shape)
    for i in range(len(shape)):
        t_shape[i] = shape[dst_order[i]]
    return t_shape


def to_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def separate_params(mace_model):
    tensors = mace_model.tensors
    params = mace_pb2.NetDef()
    params.tensors.extend(tensors)

    model = mace_model
    del model.tensors[:]
    return model, params


def convert(conf, output):
    if not os.path.exists(output):
        os.mkdir(output)

    for model_name, model_conf in conf["models"].items():
        model_output = output + "/" + model_name
        if not os.path.exists(model_output):
            os.mkdir(model_output)

        subgraph = model_conf["subgraphs"][0]
        del model_conf["subgraphs"]
        model_conf.update(subgraph)

        model_file = util.download_or_get_file(model_conf["model_file_path"],
                                               model_conf[
                                                   "model_sha256_checksum"],
                                               model_output)
        model_conf["model_file_path"] = model_file
        if "weight_file_path" in model_conf:
            weight_file = util.download_or_get_file(
                model_conf["weight_file_path"],
                model_conf["weight_sha256_checksum"], model_output)
            model_conf["weight_file_path"] = weight_file
        # TODO: remove the following after quantize tool is made
        if "quantize_range_file" in model_conf:
            range_file = util.download_or_get_file(
                model_conf["quantize_range_file"],
                "", model_output)
            model_conf["quantize_range_file"] = range_file

        mace_model = convert_model(model_conf)

        try:
            visualizer = visualize_model.ModelVisualizer(model_name,
                                                         mace_model,
                                                         model_output)
            visualizer.save_html()
        except:  # noqa
            print("Failed to visualize model:", sys.exc_info()[0])

        model, params = merge_params(mace_model)

        output_model_file = model_output + "/" + model_name + ".pb"
        output_params_file = model_output + "/" + model_name + ".data"
        with open(output_model_file, "wb") as f:
            f.write(model.SerializeToString())
        with open(output_params_file, "wb") as f:
            f.write(bytearray(params))
        with open(output_model_file + "_txt", "w") as f:
            f.write(str(model))


def convert_model(conf):
    print(conf)
    platform = conf["platform"]
    mace_check(platform in ['tensorflow', 'caffe', 'onnx'],
               "platform not supported")
    runtime = conf["runtime"]
    mace_check(
        runtime in ['cpu', 'gpu', 'dsp', 'hta', 'apu', 'cpu+gpu'],
        "runtime not supported")

    option = cvt.ConverterOption()
    if "graph_optimize_options" in conf:
        option.transformer_option = conf["graph_optimize_options"].split(',')
    option.winograd = conf.get("winograd", 0)
    option.quantize = bool(conf.get("quantize", 0))
    option.quantize_large_weights = bool(conf.get("quantize_large_weights", 0))
    option.quantize_range_file = conf.get("quantize_range_file", "")
    option.change_concat_ranges = bool(conf.get("change_concat_ranges", 0))
    option.cl_mem_type = conf.get("cl_mem_type", "image")
    option.device = device_type_map[conf.get("runtime", "cpu")]
    option.data_type = parse_data_type(conf.get("data_type", "fp16_fp32"),
                                       option.quantize)
    input_tensors = to_list(conf["input_tensors"])
    input_shapes = [parse_int_array_from_str(shape) for shape in
                    to_list(conf["input_shapes"])]
    mace_check(len(input_tensors) == len(input_shapes),
               "input node count and shape count do not match")
    input_count = len(input_tensors)
    input_data_types = [data_type_map[dt] for dt in
                        to_list(conf.get("input_data_types",
                                         ["float32"] * input_count))]
    input_data_formats = [data_format_map[df] for df in
                          to_list(conf.get("input_data_formats",
                                           ["NHWC"] * input_count))]
    input_ranges = [parse_float_array_from_str(r) for r in
                    to_list(conf.get("input_ranges",
                                     ["-1.0,1.0"] * input_count))]
    for i in range(len(input_tensors)):
        input_node = cvt.NodeInfo()
        input_node.name = input_tensors[i]
        input_node.shape = input_shapes[i]
        input_node.data_type = input_data_types[i]
        input_node.data_format = input_data_formats[i]
        if (input_node.data_format == cvt.DataFormat.NCHW and len(
              input_node.shape) == 4):
            input_node.shape = transpose_shape(input_node.shape, [0, 2, 3, 1])
            input_node.data_format = cvt.DataFormat.NHWC
        input_node.range = input_ranges[i]
        option.add_input_node(input_node)

    output_tensors = to_list(conf["output_tensors"])
    output_shapes = [parse_int_array_from_str(shape) for shape in
                     to_list(conf["output_shapes"])]
    mace_check(len(output_tensors) == len(output_shapes),
               "output node count and shape count do not match")
    output_count = len(output_tensors)
    output_data_types = [data_type_map[dt] for dt in
                         to_list(conf.get("output_data_types",
                                          ["float32"] * output_count))]
    output_data_formats = [data_format_map[df] for df in
                           to_list(conf.get("output_data_formats",
                                            ["NHWC"] * output_count))]
    for i in range(len(output_tensors)):
        output_node = cvt.NodeInfo()
        output_node.name = output_tensors[i]
        output_node.shape = output_shapes[i]
        output_node.data_type = output_data_types[i]
        output_node.data_format = output_data_formats[i]
        if output_node.data_format == cvt.DataFormat.NCHW and len(
                output_node.shape) == 4:
            output_node.shape = transpose_shape(output_node.shape,
                                                [0, 2, 3, 1])
            output_node.data_format = cvt.DataFormat.NHWC
        option.add_output_node(output_node)

    if "check_tensors" in conf:
        check_tensors = to_list(conf["check_tensors"])
        check_tensors_shapes = [parse_int_array_from_str(shape) for shape in
                                to_list(conf["check_shapes"])]
        mace_check(len(check_tensors) == len(check_tensors_shapes),
                   "check tensors count and shape count do not match.")
        for i in range(len(check_tensors)):
            check_node = cvt.NodeInfo()
            check_node.name = check_tensors[i]
            check_node.shape = check_tensors_shapes[i]
            option.add_check_node(check_node)
    else:
        option.check_nodes = option.output_nodes

    option.build()

    print("Transform model to one that can better run on device")

    if platform == 'tensorflow':
        from transform import tensorflow_converter
        converter = tensorflow_converter.TensorflowConverter(
            option, conf["model_file_path"])
    elif platform == 'caffe':
        from transform import caffe_converter
        converter = caffe_converter.CaffeConverter(option,
                                                   conf["model_file_path"],
                                                   conf["weight_file_path"])
    elif platform == 'onnx':
        from transform import onnx_converter
        converter = onnx_converter.OnnxConverter(option,
                                                 conf["model_file_path"])
    else:
        mace_check(False, "Mace do not support platorm %s yet." % platform)

    output_graph_def = converter.run()
    mace_transformer = transformer.Transformer(
        option, output_graph_def)
    output_graph_def, quantize_activation_info = mace_transformer.run()

    if option.device in [cvt.DeviceType.HEXAGON.value,
                         cvt.DeviceType.HTA.value]:
        from transform import hexagon_converter
        converter = hexagon_converter.HexagonConverter(
            option, output_graph_def, quantize_activation_info)
        output_graph_def = converter.run()
    elif runtime == 'apu':
        mace_check(platform == "tensorflow",
                   "apu only support model from tensorflow")
        from transform import apu_converter
        converter = apu_converter.ApuConverter(
            option, output_graph_def, quantize_activation_info)
        output_graph_def = converter.run()

    return output_graph_def


def merge_params(net_def):
    def tensor_to_bytes(tensor):
        if tensor.data_type == mace_pb2.DT_HALF:
            data = bytearray(
                np.array(tensor.float_data).astype(np.float16).tobytes())
            tensor.data_size = len(tensor.float_data)
        elif tensor.data_type == mace_pb2.DT_FLOAT:
            data = bytearray(
                np.array(tensor.float_data).astype(np.float32).tobytes())
            tensor.data_size = len(tensor.float_data)
        elif tensor.data_type == mace_pb2.DT_INT32:
            data = bytearray(
                np.array(tensor.int32_data).astype(np.int32).tobytes())
            tensor.data_size = len(tensor.int32_data)
        elif tensor.data_type == mace_pb2.DT_UINT8:
            data = bytearray(
                np.array(tensor.int32_data).astype(np.uint8).tolist())
            tensor.data_size = len(tensor.int32_data)
        elif tensor.data_type == mace_pb2.DT_FLOAT16:
            data = bytearray(
                np.array(tensor.float_data).astype(np.float16).tobytes())
            tensor.data_size = len(tensor.float_data)
        else:
            raise Exception('Tensor data type %s not supported' %
                            tensor.data_type)
        return data

    model_data = []
    offset = 0
    for tensor in net_def.tensors:
        raw_data = tensor_to_bytes(tensor)
        if tensor.data_type != mace_pb2.DT_UINT8 and offset % 4 != 0:
            padding = 4 - offset % 4
            model_data.extend(bytearray([0] * padding))
            offset += padding

        tensor.offset = offset
        model_data.extend(raw_data)
        offset += len(raw_data)

    for tensor in net_def.tensors:
        if tensor.data_type == mace_pb2.DT_FLOAT \
                or tensor.data_type == mace_pb2.DT_HALF \
                or tensor.data_type == mace_pb2.DT_FLOAT16:
            del tensor.float_data[:]
        elif tensor.data_type == mace_pb2.DT_INT32:
            del tensor.int32_data[:]
        elif tensor.data_type == mace_pb2.DT_UINT8:
            del tensor.int32_data[:]

    return net_def, model_data


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
        default=".",
        help="output dir")
    flgs, _ = parser.parse_known_args()
    return flgs


if __name__ == '__main__':
    flags = parse_args()
    conf = config_parser.parse(flags.config)
    convert(conf, flags.output)
