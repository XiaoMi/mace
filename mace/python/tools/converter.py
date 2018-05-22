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

import argparse
import sys
import hashlib
import os.path
import copy

from mace.proto import mace_pb2
from mace.python.tools import tf_dsp_converter_lib
from mace.python.tools import memory_optimizer
from mace.python.tools import source_converter_lib
from mace.python.tools import tensor_util
from mace.python.tools.converter_tool import base_converter as cvt
from mace.python.tools.converter_tool import tensorflow_converter
from mace.python.tools.converter_tool import caffe_converter
from mace.python.tools.converter_tool import transformer
from mace.python.tools.convert_util import mace_check


# ./bazel-bin/mace/python/tools/tf_converter --model_file quantized_test.pb \
#                                            --output quantized_test_dsp.pb \
#                                            --runtime dsp \
#                                            --input_dim input_node,1,28,28,3

FLAGS = None

device_type_map = {'cpu': cvt.DeviceType.CPU.value,
                   'gpu': cvt.DeviceType.GPU.value,
                   'dsp': cvt.DeviceType.HEXAGON.value}
device_data_type_map = {
    cvt.DeviceType.CPU.value: mace_pb2.DT_FLOAT,
    cvt.DeviceType.GPU.value: mace_pb2.DT_HALF,
    cvt.DeviceType.HEXAGON.value: mace_pb2.DT_UINT8
}


def file_checksum(fname):
    hash_func = hashlib.sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def parse_int_array_from_str(ints_str):
    return [int(int_str) for int_str in ints_str.split(',')]


def main(unused_args):
    if not os.path.isfile(FLAGS.model_file):
        print("Input graph file '" + FLAGS.model_file + "' does not exist!")
        sys.exit(-1)

    model_checksum = file_checksum(FLAGS.model_file)
    if FLAGS.model_checksum != "" and FLAGS.model_checksum != model_checksum:
        print("Model checksum mismatch: %s != %s" % (model_checksum,
                                                     FLAGS.model_checksum))
        sys.exit(-1)

    weight_checksum = None
    if FLAGS.platform == 'caffe':
        if not os.path.isfile(FLAGS.weight_file):
            print("Input weight file '" + FLAGS.weight_file +
                  "' does not exist!")
            sys.exit(-1)

        weight_checksum = file_checksum(FLAGS.weight_file)
        if FLAGS.weight_checksum != "" and \
                FLAGS.weight_checksum != weight_checksum:
            print("Weight checksum mismatch: %s != %s" %
                  (weight_checksum, FLAGS.weight_checksum))
            sys.exit(-1)

    if FLAGS.platform not in ['tensorflow', 'caffe']:
        print ("platform %s is not supported." % FLAGS.platform)
        sys.exit(-1)
    if FLAGS.runtime not in ['cpu', 'gpu', 'dsp', '']:
        print ("runtime %s is not supported." % FLAGS.runtime)
        sys.exit(-1)

    if FLAGS.runtime == 'dsp':
        if FLAGS.platform == 'tensorflow':
            output_graph_def = tf_dsp_converter_lib.convert_to_mace_pb(
                FLAGS.model_file, FLAGS.input_node, FLAGS.output_node,
                FLAGS.dsp_mode)
        else:
            print("%s does not support dsp runtime yet." % FLAGS.platform)
            sys.exit(-1)
    else:
        option = cvt.ConverterOption()
        option.winograd_enabled = bool(FLAGS.winograd)

        input_node_names = FLAGS.input_node.split(',')
        input_node_shapes = FLAGS.input_shape.split(':')
        if len(input_node_names) != len(input_node_shapes):
            raise Exception('input node count and shape count do not match.')
        for i in xrange(len(input_node_names)):
            input_node = cvt.NodeInfo()
            input_node.name = input_node_names[i]
            input_node.shape = parse_int_array_from_str(FLAGS.input_shape)
            option.add_input_node(input_node)

        output_node_names = FLAGS.output_node.split(',')
        for i in xrange(len(output_node_names)):
            output_node = cvt.NodeInfo()
            output_node.name = output_node_names[i]
            option.add_output_node(output_node)

        print("Convert model to mace model.")
        if FLAGS.platform == 'tensorflow':
            converter = tensorflow_converter.TensorflowConverter(
                option, FLAGS.model_file)
        elif FLAGS.platform == 'caffe':
            converter = caffe_converter.CaffeConverter(option,
                                                       FLAGS.model_file,
                                                       FLAGS.weight_file)

        output_graph_def = converter.run()
        print("Transform model to one that can better run on device")
        if not FLAGS.runtime:
            cpu_graph_def = copy.deepcopy(output_graph_def)
            option.device = cvt.DeviceType.CPU.value
            option.data_type = device_data_type_map[cvt.DeviceType.CPU.value]
            option.disable_transpose_filters()
            mace_cpu_transformer = transformer.Transformer(
                option, cpu_graph_def)
            cpu_graph_def = mace_cpu_transformer.run()
            print "start optimize cpu memory."
            memory_optimizer.optimize_cpu_memory(cpu_graph_def)
            print "CPU memory optimization done."

            option.device = cvt.DeviceType.GPU.value
            option.data_type = device_data_type_map[cvt.DeviceType.GPU.value]
            option.enable_transpose_filters()
            mace_gpu_transformer = transformer.Transformer(
                option, output_graph_def)
            output_gpu_graph_def = mace_gpu_transformer.run()
            print "start optimize gpu memory."
            memory_optimizer.optimize_gpu_memory(output_gpu_graph_def)
            print "GPU memory optimization done."

            print "Merge cpu and gpu ops together"
            output_graph_def.op.extend(cpu_graph_def.op)
            output_graph_def.mem_arena.mem_block.extend(
                cpu_graph_def.mem_arena.mem_block)
            print "Merge done"
        else:
            option.device = device_type_map[FLAGS.runtime]
            option.data_type = device_data_type_map[option.device]
            mace_transformer = transformer.Transformer(
                option, output_graph_def)
            output_graph_def = mace_transformer.run()

            print "start optimize memory."
            if FLAGS.runtime == 'gpu':
                memory_optimizer.optimize_gpu_memory(output_graph_def)
            elif FLAGS.runtime == 'cpu':
                memory_optimizer.optimize_cpu_memory(output_graph_def)
            else:
                mace_check(False, "runtime only support [gpu|cpu|dsp]")

            print "Memory optimization done."

    if FLAGS.obfuscate:
        tensor_util.obfuscate_name(output_graph_def)
    else:
        tensor_util.rename_tensor(output_graph_def)

    tensor_infos, model_data = tensor_util.get_tensor_info_and_model_data(
            output_graph_def, FLAGS.runtime)

    source_converter_lib.convert_to_source(
            output_graph_def, model_checksum, weight_checksum, FLAGS.template,
            FLAGS.obfuscate, FLAGS.model_tag, FLAGS.codegen_output,
            FLAGS.runtime, FLAGS.embed_model_data, FLAGS.winograd,
            FLAGS.model_load_type, tensor_infos, model_data)

    if not FLAGS.embed_model_data:
        output_dir = os.path.dirname(FLAGS.codegen_output) + '/'
        with open(output_dir + FLAGS.model_tag + '.data', "wb") as f:
            f.write(bytearray(model_data))

    if FLAGS.model_load_type == 'pb':
        tensor_util.del_tensor_data(output_graph_def, FLAGS.runtime)
        with open(FLAGS.pb_output, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        # with open(FLAGS.pb_output + '_txt', "wb") as f:
        #     # output_graph_def.ClearField('tensors')
        #     f.write(str(output_graph_def))
    print("Model conversion is completed.")


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
        "--codegen_output",
        type=str,
        default="",
        help="File to save the output graph to.")
    parser.add_argument(
        "--pb_output",
        type=str,
        default="",
        help="File to save the mace model to.")
    parser.add_argument(
        "--runtime", type=str, default="", help="Runtime: cpu/gpu/dsp")
    parser.add_argument(
        "--input_node",
        type=str,
        default="input_node",
        help="e.g., input_node")
    parser.add_argument(
        "--output_node", type=str, default="softmax", help="e.g., softmax")
    parser.add_argument(
        "--output_type", type=str, default="pb", help="output type: source/pb")
    parser.add_argument(
        "--template", type=str, default="", help="template path")
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
        type=str2bool,
        nargs='?',
        const=False,
        default=False,
        help="open winograd convolution or not")
    parser.add_argument(
        "--dsp_mode", type=int, default=0, help="dsp run mode, defalut=0")
    parser.add_argument(
        "--input_shape", type=str, default="", help="input shape.")
    parser.add_argument(
        "--platform", type=str, default="tensorflow", help="tensorflow/caffe")
    parser.add_argument(
        "--embed_model_data",
        type=str2bool,
        default=True,
        help="embed model data.")
    parser.add_argument(
        "--model_load_type",
        type=str,
        default="source",
        help="[source|pb] Load models in generated `source` code" +
                "or `pb` file.")
    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    main(unused_args=[sys.argv[0]] + unparsed)
