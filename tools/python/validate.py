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

import os
import os.path
import numpy as np
import six

from py_proto import mace_pb2
from utils import util
from utils.config_parser import DataFormat
from utils.config_parser import Platform

VALIDATION_MODULE = 'VALIDATION'


def load_data(file, data_type=mace_pb2.DT_FLOAT):
    if os.path.isfile(file):
        if data_type == mace_pb2.DT_FLOAT:
            return np.fromfile(file=file, dtype=np.float32)
        elif data_type == mace_pb2.DT_INT32:
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


def compare_output(output_name, mace_out_value,
                   out_value, validation_threshold, log_file):
    if mace_out_value.size != 0:
        pixel_accuracy = calculate_pixel_accuracy(out_value, mace_out_value)
        out_value = out_value.reshape(-1)
        mace_out_value = mace_out_value.reshape(-1)
        assert len(out_value) == len(mace_out_value)
        sqnr = calculate_sqnr(out_value, mace_out_value)
        similarity = calculate_similarity(out_value, mace_out_value)
        util.MaceLogger.summary(
            output_name + ' MACE VS training platform'
            + ' similarity: ' + str(similarity) + ' , sqnr: ' + str(sqnr)
            + ' , pixel_accuracy: ' + str(pixel_accuracy))
        if log_file:
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.write('output_name,similarity,sqnr,pixel_accuracy\n')
            summary = '{output_name},{similarity},{sqnr},{pixel_accuracy}\n' \
                .format(output_name=output_name,
                        similarity=similarity,
                        sqnr=sqnr,
                        pixel_accuracy=pixel_accuracy)
            with open(log_file, "a") as f:
                f.write(summary)
        elif similarity > validation_threshold:
            util.MaceLogger.summary(
                util.StringFormatter.block("Similarity Test Passed"))
        else:
            util.MaceLogger.error(
                util.StringFormatter.block("Similarity Test Failed"))
    else:
        util.MaceLogger.error(
            "", util.StringFormatter.block(
                "Similarity Test failed because of empty output"))


def normalize_tf_tensor_name(name):
    if name.find(':') == -1:
        return name + ':0'
    else:
        return name


def get_data_type_by_value(value):
    data_type = value.dtype
    if data_type == np.float32:
        return mace_pb2.DT_FLOAT
    elif data_type == np.int32:
        return mace_pb2.DT_INT32
    else:
        return mace_pb2.DT_FLOAT


def validate_with_file(output_names, output_shapes,
                       mace_out_file, validation_outputs_data,
                       validation_threshold, log_file):
    for i in range(len(output_names)):
        if validation_outputs_data[i].startswith("http://") or \
                validation_outputs_data[i].startswith("https://"):
            validation_file_name = util.formatted_file_name(
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
        output_file_name = util.formatted_file_name(
            mace_out_file, output_names[i])
        mace_out_value = load_data(output_file_name)
        compare_output(output_names[i], mace_out_value,
                       value, validation_threshold, log_file)


def validate_tf_model(model_file,
                      input_file, mace_out_file,
                      input_names, input_shapes, input_data_formats,
                      output_names, output_shapes, output_data_formats,
                      validation_threshold, input_data_types, log_file):
    import tensorflow as tf
    if not os.path.isfile(model_file):
        util.MaceLogger.error(
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
                        util.formatted_file_name(input_file, input_names[i]),
                        input_data_types[i])
                    input_value = input_value.reshape(input_shapes[i])
                    if input_data_formats[i] == DataFormat.NCHW and \
                            len(input_shapes[i]) == 4:
                        input_value = input_value.transpose((0, 2, 3, 1))
                    elif input_data_formats[i] == DataFormat.OIHW and \
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
                    output_file_name = util.formatted_file_name(
                        mace_out_file, output_names[i])
                    mace_out_value = load_data(
                        output_file_name,
                        get_data_type_by_value(output_values[i]))
                    if output_data_formats[i] == DataFormat.NCHW and \
                            len(output_shapes[i]) == 4:
                        mace_out_value = mace_out_value. \
                            reshape(output_shapes[i]).transpose((0, 2, 3, 1))
                    compare_output(output_names[i],
                                   mace_out_value, output_values[i],
                                   validation_threshold, log_file)


def validate_caffe_model(model_file, input_file,
                         mace_out_file, weight_file,
                         input_names, input_shapes, input_data_formats,
                         output_names, output_shapes, output_data_formats,
                         validation_threshold, log_file):
    os.environ['GLOG_minloglevel'] = '1'  # suprress Caffe verbose prints
    import caffe
    if not os.path.isfile(model_file):
        util.MaceLogger.error(
            VALIDATION_MODULE,
            "Input graph file '" + model_file + "' does not exist!")
    if not os.path.isfile(weight_file):
        util.MaceLogger.error(
            VALIDATION_MODULE,
            "Input weight file '" + weight_file + "' does not exist!")

    caffe.set_mode_cpu()

    net = caffe.Net(model_file, caffe.TEST, weights=weight_file)

    for i in range(len(input_names)):
        input_value = load_data(
            util.formatted_file_name(input_file, input_names[i]))
        input_value = input_value.reshape(input_shapes[i])
        if input_data_formats[i] == DataFormat.NHWC and \
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
        output_file_name = util.formatted_file_name(
            mace_out_file, output_names[i])
        mace_out_value = load_data(output_file_name)
        if output_data_formats[i] == DataFormat.NHWC and \
                len(output_shapes[i]) == 4:
            mace_out_value = mace_out_value.reshape(output_shapes[i]) \
                .transpose((0, 3, 1, 2))
        compare_output(output_names[i], mace_out_value,
                       value, validation_threshold, log_file)


def validate_onnx_model(model_file,
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
        util.MaceLogger.error(
            VALIDATION_MODULE,
            "onnx backend framwork '" + backend + "' is invalid.")
    if not os.path.isfile(model_file):
        util.MaceLogger.error(
            VALIDATION_MODULE,
            "Input graph file '" + model_file + "' does not exist!")
    model = onnx.load(model_file)
    input_dict = {}
    for i in range(len(input_names)):
        input_value = load_data(util.formatted_file_name(input_file,
                                                         input_names[i]),
                                input_data_types[i])
        input_value = input_value.reshape(input_shapes[i])
        if input_data_formats[i] == DataFormat.NHWC and \
                len(input_shapes[i]) == 4:
            input_value = input_value.transpose((0, 3, 1, 2))
        input_dict[input_names[i]] = input_value
    onnx_outputs = []
    for i in range(len(output_names)):
        out_shape = output_shapes[i][:]
        if output_data_formats[i] == DataFormat.NHWC and \
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
        output_file_name = util.formatted_file_name(mace_out_file,
                                                    output_names[i])
        mace_out_value = load_data(output_file_name)
        if output_data_formats[i] == DataFormat.NHWC and \
                len(output_shapes[i]) == 4:
            mace_out_value = mace_out_value.reshape(output_shapes[i]) \
                .transpose((0, 3, 1, 2))
        compare_output(output_names[i],
                       mace_out_value, value,
                       validation_threshold, log_file)


def validate(platform, model_file, weight_file, input_file, mace_out_file,
             input_shape, output_shape, input_data_format,
             output_data_format, input_node, output_node,
             validation_threshold, input_data_type, backend,
             validation_outputs_data, log_file):
    if not isinstance(validation_outputs_data, list):
        if os.path.isfile(validation_outputs_data):
            validation_outputs = [validation_outputs_data]
        else:
            validation_outputs = []
    else:
        validation_outputs = validation_outputs_data
    if validation_outputs:
        validate_with_file(output_node, output_shape,
                           mace_out_file, validation_outputs,
                           validation_threshold, log_file)
    elif platform == Platform.TENSORFLOW:
        validate_tf_model(model_file, input_file, mace_out_file,
                          input_node, input_shape, input_data_format,
                          output_node, output_shape, output_data_format,
                          validation_threshold, input_data_type,
                          log_file)
    elif platform == Platform.CAFFE:
        validate_caffe_model(model_file,
                             input_file, mace_out_file, weight_file,
                             input_node, input_shape, input_data_format,
                             output_node, output_shape, output_data_format,
                             validation_threshold, log_file)
    elif platform == Platform.ONNX:
        validate_onnx_model(model_file,
                            input_file, mace_out_file,
                            input_node, input_shape, input_data_format,
                            output_node, output_shape, output_data_format,
                            validation_threshold,
                            input_data_type, backend, log_file)
