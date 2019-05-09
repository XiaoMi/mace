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

import argparse
import sys
import hashlib
import os.path
import copy

import six

from mace.proto import mace_pb2
from mace.python.tools import model_saver
from mace.python.tools.converter_tool import base_converter as cvt
from mace.python.tools.converter_tool import transformer
from mace.python.tools.convert_util import mace_check
from mace.python.tools.visualization import visualize_model

# ./bazel-bin/mace/python/tools/tf_converter --model_file quantized_test.pb \
#                                            --output quantized_test_dsp.pb \
#                                            --runtime dsp \
#                                            --input_dim input_node,1,28,28,3

FLAGS = None

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


def parse_data_type(data_type, device_type):
    if device_type == cvt.DeviceType.CPU.value or \
            device_type == cvt.DeviceType.GPU.value:
        if data_type == 'fp32_fp32':
            return mace_pb2.DT_FLOAT
        else:
            return mace_pb2.DT_HALF
    elif device_type == cvt.DeviceType.HEXAGON.value or \
            device_type == cvt.DeviceType.HTA.value:
        return mace_pb2.DT_FLOAT
    elif device_type == cvt.DeviceType.APU.value:
        return mace_pb2.DT_FLOAT
    else:
        print("Invalid device type: " + str(device_type))


def file_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


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


def main(unused_args):
    if not os.path.isfile(FLAGS.model_file):
        six.print_("Input graph file '" +
                   FLAGS.model_file +
                   "' does not exist!", file=sys.stderr)
        sys.exit(-1)

    model_checksum = file_checksum(FLAGS.model_file)
    if FLAGS.model_checksum != "" and FLAGS.model_checksum != model_checksum:
        six.print_("Model checksum mismatch: %s != %s" %
                   (model_checksum, FLAGS.model_checksum), file=sys.stderr)
        sys.exit(-1)

    weight_checksum = None
    if FLAGS.platform == 'caffe':
        if not os.path.isfile(FLAGS.weight_file):
            six.print_("Input weight file '" + FLAGS.weight_file +
                       "' does not exist!", file=sys.stderr)
            sys.exit(-1)

        weight_checksum = file_checksum(FLAGS.weight_file)
        if FLAGS.weight_checksum != "" and \
                FLAGS.weight_checksum != weight_checksum:
            six.print_("Weight checksum mismatch: %s != %s" %
                       (weight_checksum, FLAGS.weight_checksum),
                       file=sys.stderr)
            sys.exit(-1)

    if FLAGS.platform not in ['tensorflow', 'caffe', 'onnx']:
        six.print_("platform %s is not supported." % FLAGS.platform,
                   file=sys.stderr)
        sys.exit(-1)
    if FLAGS.runtime not in ['cpu', 'gpu', 'dsp', 'hta', 'apu', 'cpu+gpu']:
        six.print_("runtime %s is not supported." % FLAGS.runtime,
                   file=sys.stderr)
        sys.exit(-1)

    option = cvt.ConverterOption()
    if FLAGS.graph_optimize_options:
        option.transformer_option = FLAGS.graph_optimize_options.split(',')
    option.winograd = FLAGS.winograd
    option.quantize = FLAGS.quantize
    option.quantize_large_weights = FLAGS.quantize_large_weights
    option.quantize_range_file = FLAGS.quantize_range_file
    option.change_concat_ranges = FLAGS.change_concat_ranges
    option.cl_mem_type = FLAGS.cl_mem_type
    option.device = device_type_map[FLAGS.runtime]
    option.data_type = parse_data_type(FLAGS.data_type, option.device)

    input_node_names = FLAGS.input_node.split(',')
    input_data_types = FLAGS.input_data_types.split(',')
    input_node_shapes = FLAGS.input_shape.split(':')
    input_node_formats = FLAGS.input_data_formats.split(",")
    if FLAGS.input_range:
        input_node_ranges = FLAGS.input_range.split(':')
    else:
        input_node_ranges = []
    if len(input_node_names) != len(input_node_shapes):
        raise Exception('input node count and shape count do not match.')
    for i in six.moves.range(len(input_node_names)):
        input_node = cvt.NodeInfo()
        input_node.name = input_node_names[i]
        input_node.data_type = data_type_map[input_data_types[i]]
        input_node.data_format = data_format_map[input_node_formats[i]]
        input_node.shape = parse_int_array_from_str(input_node_shapes[i])
        if input_node.data_format == cvt.DataFormat.NCHW and\
                len(input_node.shape) == 4:
            input_node.shape = transpose_shape(input_node.shape, [0, 2, 3, 1])
            input_node.data_format = cvt.DataFormat.NHWC
        if len(input_node_ranges) > i:
            input_node.range = parse_float_array_from_str(input_node_ranges[i])
        option.add_input_node(input_node)

    output_node_names = FLAGS.output_node.split(',')
    output_data_types = FLAGS.output_data_types.split(',')
    output_node_shapes = FLAGS.output_shape.split(':')
    output_node_formats = FLAGS.output_data_formats.split(",")
    if len(output_node_names) != len(output_node_shapes):
        raise Exception('output node count and shape count do not match.')
    for i in six.moves.range(len(output_node_names)):
        output_node = cvt.NodeInfo()
        output_node.name = output_node_names[i]
        output_node.data_type = data_type_map[output_data_types[i]]
        output_node.data_format = data_format_map[output_node_formats[i]]
        output_node.shape = parse_int_array_from_str(output_node_shapes[i])
        if output_node.data_format == cvt.DataFormat.NCHW and\
                len(output_node.shape) == 4:
            output_node.shape = transpose_shape(output_node.shape,
                                                [0, 2, 3, 1])
            output_node.data_format = cvt.DataFormat.NHWC
        option.add_output_node(output_node)

    if FLAGS.check_node != '':
        check_node_names = FLAGS.check_node.split(',')
        check_node_shapes = FLAGS.check_shape.split(':')
        if len(check_node_names) != len(check_node_shapes):
            raise Exception('check node count and shape count do not match.')
        for i in six.moves.range(len(check_node_names)):
            check_node = cvt.NodeInfo()
            check_node.name = check_node_names[i]
            check_node.shape = parse_int_array_from_str(check_node_shapes[i])
            option.add_check_node(check_node)
    else:
        option.check_nodes = option.output_nodes

    option.build()

    print("Transform model to one that can better run on device")
    if FLAGS.platform == 'tensorflow':
        from mace.python.tools.converter_tool import tensorflow_converter
        converter = tensorflow_converter.TensorflowConverter(
            option, FLAGS.model_file)
    elif FLAGS.platform == 'caffe':
        from mace.python.tools.converter_tool import caffe_converter
        converter = caffe_converter.CaffeConverter(option,
                                                   FLAGS.model_file,
                                                   FLAGS.weight_file)
    elif FLAGS.platform == 'onnx':
        from mace.python.tools.converter_tool import onnx_converter
        converter = onnx_converter.OnnxConverter(option, FLAGS.model_file)
    else:
        six.print_("Mace do not support platorm %s yet." % FLAGS.platform,
                   file=sys.stderr)
        exit(1)

    output_graph_def = converter.run()
    mace_transformer = transformer.Transformer(
        option, output_graph_def)
    output_graph_def, quantize_activation_info = mace_transformer.run()

    if option.device in [cvt.DeviceType.HEXAGON.value,
                         cvt.DeviceType.HTA.value]:
        from mace.python.tools.converter_tool import hexagon_converter
        converter = hexagon_converter.HexagonConverter(
            option, output_graph_def, quantize_activation_info)
        output_graph_def = converter.run()
    elif FLAGS.runtime == 'apu':
        if FLAGS.platform != 'tensorflow':
            raise Exception('apu only support model from tensorflow')
        from mace.python.tools.converter_tool import apu_converter
        converter = apu_converter.ApuConverter(
            option, output_graph_def, quantize_activation_info)
        output_graph_def = converter.run()

    try:
        visualizer = visualize_model.ModelVisualizer(FLAGS.model_tag,
                                                     output_graph_def)
        visualizer.save_html()
    except:  # noqa
        print("Failed to visualize model:", sys.exc_info()[0])

    model_saver.save_model(
        option, output_graph_def, model_checksum, weight_checksum,
        FLAGS.template_dir, FLAGS.obfuscate, FLAGS.model_tag,
        FLAGS.output_dir,
        FLAGS.embed_model_data,
        FLAGS.winograd,
        FLAGS.model_graph_format)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="TensorFlow \'GraphDef\' file to load, "
             "Onnx model file .onnx to load, "
             "Caffe prototxt file to load.")
    parser.add_argument(
        "--weight_file", type=str, default="", help="Caffe data file to load.")
    parser.add_argument(
        "--model_checksum",
        type=str,
        default="",
        help="Model file sha256 checksum")
    parser.add_argument(
        "--weight_checksum",
        type=str,
        default="",
        help="Weight file sha256 checksum")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="File to save the output graph to.")
    parser.add_argument(
        "--runtime", type=str, default="", help="Runtime: cpu/gpu/dsp/apu")
    parser.add_argument(
        "--input_node",
        type=str,
        default="input_node",
        help="e.g., input_node")
    parser.add_argument(
        "--input_data_types",
        type=str,
        default="float32",
        help="e.g., float32|int32")
    parser.add_argument(
        "--input_data_formats",
        type=str,
        default="NHWC",
        help="e.g., NHWC,NONE")
    parser.add_argument(
        "--output_node", type=str, default="softmax", help="e.g., softmax")
    parser.add_argument(
        "--output_data_types",
        type=str,
        default="float32",
        help="e.g., float32|int32")
    parser.add_argument(
        "--output_data_formats",
        type=str,
        default="NHWC",
        help="e.g., NHWC,NONE")
    parser.add_argument(
        "--check_node", type=str, default="softmax", help="e.g., softmax")
    parser.add_argument(
        "--template_dir", type=str, default="", help="template path")
    parser.add_argument(
        "--obfuscate",
        type=str2bool,
        nargs='?',
        const=False,
        default=False,
        help="obfuscate model names")
    parser.add_argument(
        "--model_tag",
        type=str,
        default="",
        help="model tag for generated function and namespace")
    parser.add_argument(
        "--winograd",
        type=int,
        default=0,
        help="Which version of winograd convolution to use. [2 | 4]")
    parser.add_argument(
        "--dsp_mode", type=int, default=0, help="dsp run mode, defalut=0")
    parser.add_argument(
        "--input_shape", type=str, default="", help="input shape.")
    parser.add_argument(
        "--input_range", type=str, default="", help="input range.")
    parser.add_argument(
        "--output_shape", type=str, default="", help="output shape.")
    parser.add_argument(
        "--check_shape", type=str, default="", help="check shape.")
    parser.add_argument(
        "--platform",
        type=str,
        default="tensorflow",
        help="tensorflow/caffe/onnx")
    parser.add_argument(
        "--embed_model_data",
        type=str2bool,
        default=True,
        help="embed model data.")
    parser.add_argument(
        "--model_graph_format",
        type=str,
        default="file",
        help="[file|code] build models to code" +
             "or `Protobuf` file.")
    parser.add_argument(
        "--data_type",
        type=str,
        default="fp16_fp32",
        help="fp16_fp32/fp32_fp32")
    parser.add_argument(
        "--graph_optimize_options",
        type=str,
        default="",
        help="graph optimize options")
    parser.add_argument(
        "--quantize",
        type=str2bool,
        nargs='?',
        const=False,
        default=False,
        help="quantize model")
    parser.add_argument(
        "--quantize_large_weights",
        type=str2bool,
        nargs='?',
        const=False,
        default=False,
        help="quantize large weights for compression")
    parser.add_argument(
        "--quantize_range_file",
        type=str,
        default="",
        help="file path of quantize range for each tensor")
    parser.add_argument(
        "--change_concat_ranges",
        type=str2bool,
        nargs='?',
        const=False,
        default=False,
        help="change ranges to use memcpy for quantized concat")
    parser.add_argument(
        "--cl_mem_type",
        type=str,
        default="image",
        help="which memory type to use.[image|buffer]")
    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)
