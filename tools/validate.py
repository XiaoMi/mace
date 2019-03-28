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
import os
import os.path
import numpy as np
import re
import six

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


def calculate_sqnr(expected, actual):
    noise = expected - actual

    def power_sum(xs):
        return sum([x * x for x in xs])

    signal_power_sum = power_sum(expected)
    noise_power_sum = power_sum(noise)
    return signal_power_sum / (noise_power_sum + 1e-15)


def calculate_similarity(u, v, data_type=np.float64):
    if u.dtype is not data_type:
        u = u.astype(data_type)
    if v.dtype is not data_type:
        v = v.astype(data_type)
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def calculate_pixel_accuracy(out_value, mace_out_value):
    if len(out_value.shape) < 2:
        return 1.0
    out_value = out_value.reshape((-1, out_value.shape[-1]))
    batches = out_value.shape[0]
    classes = out_value.shape[1]
    mace_out_value = mace_out_value.reshape((batches, classes))
    correct_count = 0
    for i in range(batches):
        if np.argmax(out_value[i]) == np.argmax(mace_out_value[i]):
            correct_count += 1
    return 1.0 * correct_count / batches


def compare_output(platform, device_type, output_name, mace_out_value,
                   out_value, validation_threshold, log_file):
    if mace_out_value.size != 0:
        pixel_accuracy = calculate_pixel_accuracy(out_value, mace_out_value)
        out_value = out_value.reshape(-1)
        mace_out_value = mace_out_value.reshape(-1)
        assert len(out_value) == len(mace_out_value)
        sqnr = calculate_sqnr(out_value, mace_out_value)
        similarity = calculate_similarity(out_value, mace_out_value)
        common.MaceLogger.summary(
            output_name + ' MACE VS ' + platform.upper()
            + ' similarity: ' + str(similarity) + ' , sqnr: ' + str(sqnr)
            + ' , pixel_accuracy: ' + str(pixel_accuracy))
        if log_file:
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.write('output_name,similarity,sqnr,pixel_accuracy\n')
            summary = '{output_name},{similarity},{sqnr},{pixel_accuracy}\n'\
                .format(output_name=output_name,
                        similarity=similarity,
                        sqnr=sqnr,
                        pixel_accuracy=pixel_accuracy)
            with open(log_file, "a") as f:
                f.write(summary)
        elif similarity > validation_threshold:
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


def validate_with_file(platform, device_type,
                       output_names, output_shapes,
                       mace_out_file, validation_outputs_data,
                       validation_threshold, log_file):
    for i in range(len(output_names)):
        if validation_outputs_data[i].startswith("http://") or \
                validation_outputs_data[i].startswith("https://"):
            validation_file_name = common.formatted_file_name(
                mace_out_file, output_names[i] + '_validation')
            six.moves.urllib.request.urlretrieve(validation_outputs_data[i],
                                                 validation_file_name)
        else:
            validation_file_name = validation_outputs_data[i]
        value = load_data(validation_file_name)
        out_shape = output_shapes[i]
        if len(out_shape) == 4:
            out_shape[1], out_shape[2], out_shape[3] = \
                out_shape[3], out_shape[1], out_shape[2]
            value = value.reshape(out_shape).transpose((0, 2, 3, 1))
        output_file_name = common.formatted_file_name(
            mace_out_file, output_names[i])
        mace_out_value = load_data(output_file_name)
        compare_output(platform, device_type, output_names[i], mace_out_value,
                       value, validation_threshold, log_file)


def validate_tf_model(platform, device_type, model_file,
                      input_file, mace_out_file,
                      input_names, input_shapes, input_data_formats,
                      output_names, output_shapes, output_data_formats,
                      validation_threshold, input_data_types, log_file):
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
                    if input_data_formats[i] == common.DataFormat.NCHW and\
                            len(input_shapes[i]) == 4:
                        input_value = input_value.transpose((0, 2, 3, 1))
                    elif input_data_formats[i] == common.DataFormat.OIHW and \
                            len(input_shapes[i]) == 4:
                        # OIHW -> HWIO
                        input_value = input_value.transpose((2, 3, 1, 0))
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
                    if output_data_formats[i] == common.DataFormat.NCHW and\
                            len(output_shapes[i]) == 4:
                        mace_out_value = mace_out_value.\
                            reshape(output_shapes[i]).transpose((0, 2, 3, 1))
                    compare_output(platform, device_type, output_names[i],
                                   mace_out_value, output_values[i],
                                   validation_threshold, log_file)


def validate_caffe_model(platform, device_type, model_file, input_file,
                         mace_out_file, weight_file,
                         input_names, input_shapes, input_data_formats,
                         output_names, output_shapes, output_data_formats,
                         validation_threshold, log_file):
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
        input_value = input_value.reshape(input_shapes[i])
        if input_data_formats[i] == common.DataFormat.NHWC and \
                len(input_shapes[i]) == 4:
            input_value = input_value.transpose((0, 3, 1, 2))
        input_blob_name = input_names[i]
        try:
            if input_names[i] in net.top_names:
                input_blob_name = net.top_names[input_names[i]][0]
        except ValueError:
            pass
        new_shape = input_value.shape
        net.blobs[input_blob_name].reshape(*new_shape)
        for index in range(input_value.shape[0]):
            net.blobs[input_blob_name].data[index] = input_value[index]

    net.forward()

    for i in range(len(output_names)):
        value = net.blobs[output_names[i]].data
        output_file_name = common.formatted_file_name(
            mace_out_file, output_names[i])
        mace_out_value = load_data(output_file_name)
        if output_data_formats[i] == common.DataFormat.NHWC and \
                len(output_shapes[i]) == 4:
            mace_out_value = mace_out_value.reshape(output_shapes[i])\
                .transpose((0, 3, 1, 2))
        compare_output(platform, device_type, output_names[i], mace_out_value,
                       value, validation_threshold, log_file)


def validate_onnx_model(platform, device_type, model_file,
                        input_file, mace_out_file,
                        input_names, input_shapes, input_data_formats,
                        output_names, output_shapes, output_data_formats,
                        validation_threshold, input_data_types,
                        backend, log_file):
    import onnx
    if backend == "tensorflow":
        from onnx_tf.backend import prepare
        print("valivate on onnx tensorflow backend.")
    elif backend == "caffe2" or backend == "pytorch":
        from caffe2.python.onnx.backend import prepare
        print("valivate on onnx caffe2 backend.")
    else:
        common.MaceLogger.error(
            VALIDATION_MODULE,
            "onnx backend framwork '" + backend + "' is invalid.")
    if not os.path.isfile(model_file):
        common.MaceLogger.error(
            VALIDATION_MODULE,
            "Input graph file '" + model_file + "' does not exist!")
    model = onnx.load(model_file)
    input_dict = {}
    for i in range(len(input_names)):
        input_value = load_data(common.formatted_file_name(input_file,
                                                           input_names[i]),
                                input_data_types[i])
        input_value = input_value.reshape(input_shapes[i])
        if input_data_formats[i] == common.DataFormat.NHWC and \
                len(input_shapes[i]) == 4:
            input_value = input_value.transpose((0, 3, 1, 2))
        input_dict[input_names[i]] = input_value
    onnx_outputs = []
    for i in range(len(output_names)):
        out_shape = output_shapes[i]
        if output_data_formats[i] == common.DataFormat.NHWC and\
                len(out_shape) == 4:
            out_shape[1], out_shape[2], out_shape[3] = \
                out_shape[3], out_shape[1], out_shape[2]
        onnx_outputs.append(
            onnx.helper.make_tensor_value_info(output_names[i],
                                               onnx.TensorProto.FLOAT,
                                               out_shape))
    model.graph.output.extend(onnx_outputs)
    rep = prepare(model)

    output_values = rep.run(input_dict)
    for i in range(len(output_names)):
        out_name = output_names[i]
        value = output_values[out_name].flatten()
        output_file_name = common.formatted_file_name(mace_out_file,
                                                      output_names[i])
        mace_out_value = load_data(output_file_name)
        if output_data_formats[i] == common.DataFormat.NHWC and \
                len(output_shapes[i]) == 4:
            mace_out_value = mace_out_value.reshape(output_shapes[i]) \
                .transpose((0, 3, 1, 2))
        compare_output(platform, device_type, output_names[i],
                       mace_out_value, value,
                       validation_threshold, log_file)


def validate(platform, model_file, weight_file, input_file, mace_out_file,
             device_type, input_shape, output_shape, input_data_format_str,
             output_data_format_str, input_node, output_node,
             validation_threshold, input_data_type, backend,
             validation_outputs_data, log_file):
    input_names = [name for name in input_node.split(',')]
    input_shape_strs = [shape for shape in input_shape.split(':')]
    input_shapes = [[int(x) for x in common.split_shape(shape)]
                    for shape in input_shape_strs]
    output_shape_strs = [shape for shape in output_shape.split(':')]
    output_shapes = [[int(x) for x in common.split_shape(shape)]
                     for shape in output_shape_strs]
    input_data_formats = [df for df in input_data_format_str.split(',')]
    output_data_formats = [df for df in output_data_format_str.split(',')]
    if input_data_type:
        input_data_types = [data_type
                            for data_type in input_data_type.split(',')]
    else:
        input_data_types = ['float32'] * len(input_names)
    output_names = [name for name in output_node.split(',')]
    assert len(input_names) == len(input_shapes)
    if not isinstance(validation_outputs_data, list):
        if os.path.isfile(validation_outputs_data):
            validation_outputs = [validation_outputs_data]
        else:
            validation_outputs = []
    else:
        validation_outputs = validation_outputs_data
    if validation_outputs:
        validate_with_file(platform, device_type, output_names, output_shapes,
                           mace_out_file, validation_outputs,
                           validation_threshold, log_file)
    elif platform == 'tensorflow':
        validate_tf_model(platform, device_type,
                          model_file, input_file, mace_out_file,
                          input_names, input_shapes, input_data_formats,
                          output_names, output_shapes, output_data_formats,
                          validation_threshold, input_data_types,
                          log_file)
    elif platform == 'caffe':
        validate_caffe_model(platform, device_type, model_file,
                             input_file, mace_out_file, weight_file,
                             input_names, input_shapes, input_data_formats,
                             output_names, output_shapes, output_data_formats,
                             validation_threshold, log_file)
    elif platform == 'onnx':
        validate_onnx_model(platform, device_type, model_file,
                            input_file, mace_out_file,
                            input_names, input_shapes, input_data_formats,
                            output_names, output_shapes, output_data_formats,
                            validation_threshold,
                            input_data_types, backend, log_file)


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
        "--input_data_format", type=str, default="NHWC",
        help="input data format.")
    parser.add_argument(
        "--output_shape", type=str, default="1,64,64,2", help="output shape.")
    parser.add_argument(
        "--output_data_format", type=str, default="NHWC",
        help="output data format.")
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
    parser.add_argument(
        "--backend",
        type=str,
        default="tensorflow",
        help="onnx backend framwork")
    parser.add_argument(
        "--validation_outputs_data", type=str,
        default="", help="validation outputs data file path.")
    parser.add_argument(
        "--log_file", type=str, default="", help="log file.")

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
             FLAGS.input_data_format,
             FLAGS.output_data_format,
             FLAGS.input_node,
             FLAGS.output_node,
             FLAGS.validation_threshold,
             FLAGS.input_data_type,
             FLAGS.backend,
             FLAGS.validation_outputs_data,
             FLAGS.log_file)
