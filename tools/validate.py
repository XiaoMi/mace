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
import os
import os.path
import numpy as np
import re
from scipy import spatial
from scipy import stats

import common

# Validation Flow:
# 1. Generate input data
# 2. Use mace_run to run model on phone.
# 3. adb pull the result.
# 4. Compare output data of mace and tf
#    python validate.py --model_file tf_model_opt.pb \
#        --input_file input_file \
#        --mace_out_file output_file \
#        --input_node input_node \
#        --output_node output_node \
#        --input_shape 1,64,64,3 \
#        --output_shape 1,64,64,2
#        --validation_threshold 0.995

VALIDATION_MODULE = 'VALIDATION'


def load_data(file, data_type='float32'):
    if os.path.isfile(file):
        if data_type == 'float32':
            return np.fromfile(file=file, dtype=np.float32)
        elif data_type == 'int32':
            return np.fromfile(file=file, dtype=np.int32)
    return np.empty([0])


def compare_output(platform, device_type, output_name, mace_out_value,
                   out_value, validation_threshold):
    if mace_out_value.size != 0:
        out_value = out_value.reshape(-1)
        mace_out_value = mace_out_value.reshape(-1)
        assert len(out_value) == len(mace_out_value)
        similarity = (1 - spatial.distance.cosine(out_value, mace_out_value))
        common.MaceLogger.summary(
            output_name + ' MACE VS ' + platform.upper()
            + ' similarity: ' + str(similarity))
        if similarity > validation_threshold:
            common.MaceLogger.summary(
                common.StringFormatter.block("Similarity Test Passed"))
        else:
            common.MaceLogger.error(
                "", common.StringFormatter.block("Similarity Test Failed"))
    else:
        common.MaceLogger.error(
            "", common.StringFormatter.block(
                "Similarity Test failed because of empty output"))


def normalize_tf_tensor_name(name):
    if name.find(':') == -1:
        return name + ':0'
    else:
        return name


def validate_tf_model(platform, device_type, model_file, input_file,
                      mace_out_file, input_names, input_shapes,
                      output_names, validation_threshold, input_data_types):
    import tensorflow as tf
    if not os.path.isfile(model_file):
        common.MaceLogger.error(
            VALIDATION_MODULE,
            "Input graph file '" + model_file + "' does not exist!")

    tf.reset_default_graph()
    input_graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)
        tf.import_graph_def(input_graph_def, name="")

        with tf.Session() as session:
            with session.graph.as_default() as graph:
                tf.import_graph_def(input_graph_def, name="")
                input_dict = {}
                for i in range(len(input_names)):
                    input_value = load_data(
                        common.formatted_file_name(input_file, input_names[i]),
                        input_data_types[i])
                    input_value = input_value.reshape(input_shapes[i])
                    input_node = graph.get_tensor_by_name(
                        normalize_tf_tensor_name(input_names[i]))
                    input_dict[input_node] = input_value

                output_nodes = []
                for name in output_names:
                    output_nodes.extend(
                        [graph.get_tensor_by_name(
                            normalize_tf_tensor_name(name))])
                output_values = session.run(output_nodes, feed_dict=input_dict)
                for i in range(len(output_names)):
                    output_file_name = common.formatted_file_name(
                        mace_out_file, output_names[i])
                    mace_out_value = load_data(output_file_name)
                    compare_output(platform, device_type, output_names[i],
                                   mace_out_value, output_values[i],
                                   validation_threshold)


def validate_caffe_model(platform, device_type, model_file, input_file,
                         mace_out_file, weight_file, input_names, input_shapes,
                         output_names, output_shapes, validation_threshold):
    os.environ['GLOG_minloglevel'] = '1'  # suprress Caffe verbose prints
    import caffe
    if not os.path.isfile(model_file):
        common.MaceLogger.error(
            VALIDATION_MODULE,
            "Input graph file '" + model_file + "' does not exist!")
    if not os.path.isfile(weight_file):
        common.MaceLogger.error(
            VALIDATION_MODULE,
            "Input weight file '" + weight_file + "' does not exist!")

    caffe.set_mode_cpu()

    net = caffe.Net(model_file, caffe.TEST, weights=weight_file)

    for i in range(len(input_names)):
        input_value = load_data(
            common.formatted_file_name(input_file, input_names[i]))
        input_value = input_value.reshape(input_shapes[i]).transpose((0, 3, 1,
                                                                      2))
        input_blob_name = input_names[i]
        try:
            if input_names[i] in net.top_names:
                input_blob_name = net.top_names[input_names[i]][0]
        except ValueError:
            pass
        net.blobs[input_blob_name].data[0] = input_value

    net.forward()

    for i in range(len(output_names)):
        value = net.blobs[net.top_names[output_names[i]][0]].data
        out_shape = output_shapes[i]
        if len(out_shape) == 4:
            out_shape[1], out_shape[2], out_shape[3] = \
                out_shape[3], out_shape[1], out_shape[2]
            value = value.reshape(out_shape).transpose((0, 2, 3, 1))
        output_file_name = common.formatted_file_name(
            mace_out_file, output_names[i])
        mace_out_value = load_data(output_file_name)
        compare_output(platform, device_type, output_names[i], mace_out_value,
                       value, validation_threshold)


def validate(platform, model_file, weight_file, input_file, mace_out_file,
             device_type, input_shape, output_shape, input_node, output_node,
             validation_threshold, input_data_type):
    input_names = [name for name in input_node.split(',')]
    input_shape_strs = [shape for shape in input_shape.split(':')]
    input_shapes = [[int(x) for x in shape.split(',')]
                    for shape in input_shape_strs]
    if input_data_type:
        input_data_types = [data_type
                            for data_type in input_data_type.split(',')]
    else:
        input_data_types = ['float32'] * len(input_names)
    output_names = [name for name in output_node.split(',')]
    assert len(input_names) == len(input_shapes)

    if platform == 'tensorflow':
        validate_tf_model(platform, device_type, model_file, input_file,
                          mace_out_file, input_names, input_shapes,
                          output_names, validation_threshold, input_data_types)
    elif platform == 'caffe':
        output_shape_strs = [shape for shape in output_shape.split(':')]
        output_shapes = [[int(x) for x in shape.split(',')]
                         for shape in output_shape_strs]
        validate_caffe_model(platform, device_type, model_file, input_file,
                             mace_out_file, weight_file, input_names,
                             input_shapes, output_names, output_shapes,
                             validation_threshold)


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform", type=str, default="", help="TensorFlow or Caffe.")
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="TensorFlow or Caffe \'GraphDef\' file to load.")
    parser.add_argument(
        "--weight_file",
        type=str,
        default="",
        help="caffe model file to load.")
    parser.add_argument(
        "--input_file", type=str, default="", help="input file.")
    parser.add_argument(
        "--mace_out_file",
        type=str,
        default="",
        help="mace output file to load.")
    parser.add_argument(
        "--device_type", type=str, default="", help="mace runtime device.")
    parser.add_argument(
        "--input_shape", type=str, default="1,64,64,3", help="input shape.")
    parser.add_argument(
        "--output_shape", type=str, default="1,64,64,2", help="output shape.")
    parser.add_argument(
        "--input_node", type=str, default="input_node", help="input node")
    parser.add_argument(
        "--input_data_type",
        type=str,
        default="",
        help="input data type")
    parser.add_argument(
        "--output_node", type=str, default="output_node", help="output node")
    parser.add_argument(
        "--validation_threshold", type=float, default=0.995,
        help="validation similarity threshold")

    return parser.parse_known_args()


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    validate(FLAGS.platform,
             FLAGS.model_file,
             FLAGS.weight_file,
             FLAGS.input_file,
             FLAGS.mace_out_file,
             FLAGS.device_type,
             FLAGS.input_shape,
             FLAGS.output_shape,
             FLAGS.input_node,
             FLAGS.output_node,
             FLAGS.validation_threshold,
             FLAGS.input_data_type)
