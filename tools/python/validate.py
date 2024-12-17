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
import sys
import os.path
import numpy as np
import six

from py_proto import mace_pb2
from utils import util
from utils.config_parser import DataFormat
from utils.config_parser import Platform
from utils.util import mace_check

VALIDATION_MODULE = 'VALIDATION'


def load_data(file, data_type=mace_pb2.DT_FLOAT):
    if os.path.isfile(file):
        if data_type == mace_pb2.DT_FLOAT or \
                data_type == mace_pb2.DT_FLOAT16 or \
                data_type == mace_pb2.DT_BFLOAT16:
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


def calculate_pixel_accuracy(out_value, mace_out_value,
                             output_shape, output_data_format):
    out_value = out_value.reshape(output_shape)
    mace_out_value = mace_out_value.reshape(output_shape)
    if len(output_shape) == 4 and output_data_format == DataFormat.NCHW:
        out_value = out_value.transpose([0, 2, 3, 1])
        mace_out_value = mace_out_value.transpose([0, 2, 3, 1])
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
                   out_value, validation_threshold, log_file,
                   output_shape, output_data_format):
    if mace_out_value.size != 0:
        pixel_accuracy = calculate_pixel_accuracy(out_value, mace_out_value,
                                                  output_shape,
                                                  output_data_format)
        out_value = out_value.reshape(-1)
        mace_out_value = mace_out_value.reshape(-1)
        assert len(out_value) == len(mace_out_value)
        sqnr = calculate_sqnr(out_value, mace_out_value)
        similarity = calculate_similarity(out_value, mace_out_value)
        '''util.MaceLogger.summary(
            output_name + ' MACE VS training platform'
            + ' similarity: ' + str(similarity) + ' , sqnr: ' + str(sqnr)
            + ' , pixel_accuracy: ' + str(pixel_accuracy))'''
        if log_file:
            with open(log_file, 'a') as f:
                f.write('%-20s%-30s%-30s%-20s' %(output_name,similarity,sqnr,pixel_accuracy))
                if similarity > validation_threshold:
                    f.write('%-20s'%'PASS'+'\n')
                else:
                    f.write('%-20s'%'NOPASS'+'\n')
        else:
            if similarity > validation_threshold:
                util.MaceLogger.summary(
                    util.StringFormatter.block("Similarity Test Passed"))
            else:
                util.MaceLogger.summary(
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


def get_real_out_value_shape_df(platform, mace_out_value, output_shape,
                                output_data_format):
    real_output_shape = output_shape
    real_output_data_format = output_data_format
    if len(output_shape) == 4:
        # These platforms use NHWC, if MACE's output is NCHW,
        # transpose the output of MACE from NCHW to NHWC.
        if output_data_format == DataFormat.NCHW and \
                platform.name.lower() in ["tensorflow", "keras"]:
            mace_out_value = mace_out_value.reshape(output_shape)\
                .transpose((0, 2, 3, 1))
            real_output_shape = list(mace_out_value.shape)
            real_output_data_format = DataFormat.NHWC
        # These platforms use NCHW, if MACE's output is NHWC,
        # transpose the output of MACE from NHWC to NCHW.
        elif output_data_format == DataFormat.NHWC and \
                platform.name.lower() in ["pytorch", "caffe", "onnx",
                                          "megengine"]:
            mace_out_value = mace_out_value.reshape(output_shape)\
                .transpose((0, 3, 1, 2))
            real_output_shape = list(mace_out_value.shape)
            real_output_data_format = DataFormat.NCHW
    return mace_out_value, real_output_shape, real_output_data_format


def validate_with_file(platform, output_names, output_shapes,
                       mace_out_file, validation_outputs_data,
                       validation_threshold, log_file,
                       output_data_formats):
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
        output_file_name = util.formatted_file_name(
            mace_out_file, output_names[i])
        mace_out_value = load_data(output_file_name)
        mace_out_value, real_output_shape, real_output_data_format = \
            get_real_out_value_shape_df(platform,
                                        mace_out_value,
                                        output_shapes[i],
                                        output_data_formats[i])
        compare_output(output_names[i], mace_out_value,
                       value, validation_threshold, log_file,
                       real_output_shape, real_output_data_format)


def validate_tf_model(platform, model_file,
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
                    mace_out_value, real_out_shape, real_out_data_format = \
                        get_real_out_value_shape_df(platform,
                                                    mace_out_value,
                                                    output_shapes[i],
                                                    output_data_formats[i])
                    compare_output(output_names[i],
                                   mace_out_value, output_values[i],
                                   validation_threshold, log_file,
                                   real_out_shape, real_out_data_format)


def validate_pytorch_model(platform, model_file,
                           input_file, mace_out_file,
                           input_names, input_shapes, input_data_formats,
                           output_names, output_shapes, output_data_formats,
                           validation_threshold, input_data_types, log_file):
    import torch
    loaded_model = torch.jit.load(model_file)
    pytorch_inputs = []
    for i in range(len(input_names)):
        input_value = load_data(
            util.formatted_file_name(input_file, input_names[i]),
            input_data_types[i])
        input_value = input_value.reshape(input_shapes[i])
        if input_data_formats[i] == DataFormat.NHWC and \
                len(input_shapes[i]) == 4:
            input_value = input_value.transpose((0, 3, 1, 2))
        input_value = torch.from_numpy(input_value)
        pytorch_inputs.append(input_value)
    with torch.no_grad():
        pytorch_outputs = loaded_model(*pytorch_inputs)

    if isinstance(pytorch_outputs, torch.Tensor):
        pytorch_outputs = [pytorch_outputs]
    else:
        if not isinstance(pytorch_outputs, (list, tuple)):
            print('return type {} unsupported'.format(type(pytorch_outputs)))
            sys.exit(1)
    for i in range(len(output_names)):
        value = pytorch_outputs[i].numpy()
        output_file_name = util.formatted_file_name(
            mace_out_file, output_names[i])
        mace_out_value = load_data(output_file_name)
        mace_out_value, real_output_shape, real_output_data_format = \
            get_real_out_value_shape_df(platform,
                                        mace_out_value,
                                        output_shapes[i],
                                        output_data_formats[i])
        compare_output(output_names[i], mace_out_value,
                       value, validation_threshold, log_file,
                       real_output_shape, real_output_data_format)


def validate_caffe_model(platform, model_file, input_file,
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
        mace_out_value, real_output_shape, real_output_data_format = \
            get_real_out_value_shape_df(platform,
                                        mace_out_value,
                                        output_shapes[i],
                                        output_data_formats[i])
        compare_output(output_names[i], mace_out_value,
                       value, validation_threshold, log_file,
                       real_output_shape, real_output_data_format)


# Remove annoying warning from onnxruntime.
def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'  # noqa
        )
        return in_path

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])


def validate_onnx_model(platform, model_file,
                        input_file, mace_out_file,
                        input_names, input_shapes, input_data_formats,
                        output_names, output_shapes, output_data_formats,
                        validation_threshold, input_data_types,
                        backend, log_file):
    print("validate on onnxruntime.")
    import onnx
    import onnxruntime as onnxrt

    if not os.path.isfile(model_file):
        util.MaceLogger.error(
            VALIDATION_MODULE,
            "Input graph file '" + model_file + "' does not exist!")

    model = onnx.load(model_file)
    remove_initializer_from_input(model)
    model_outputs = set()
    for output in model.graph.output:
        model_outputs.add(output.name)
    for output_name in output_names:
        if output_name not in model_outputs:
            layer_value_info = onnx.helper.ValueInfoProto()
            layer_value_info.name = output_name
            model.graph.output.append(layer_value_info)

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

    sess = onnxrt.InferenceSession(model.SerializeToString())
    output_values = sess.run(output_names, input_dict)
    if log_file:
        with open(log_file, 'w') as f:
            f.write('%-20s%-30s%-30s%-20s' %('output_name','similarity','sqnr','pixel_accuracy') + '\n')
    for i in range(len(output_names)):
        value = output_values[i].flatten()
        output_file_name = util.formatted_file_name(mace_out_file,
                                                    output_names[i])
        mace_out_value = load_data(output_file_name)
        real_output_shape = output_shapes[i]
        real_output_data_format = DataFormat.NONE
        mace_out_value, real_output_shape, real_output_data_format = \
            get_real_out_value_shape_df(platform,
                                        mace_out_value,
                                        output_shapes[i],
                                        output_data_formats[i])
        print(f'tensor name: {output_names[i]}, shapes: {output_shapes[i]}, data_formats: {output_data_formats[i]}')
        ##下面是用来生成一个tensor分别在mace和nn库上跑完的具体数值文件
        tensor_names = ['1669','1670','1671','1674','1689','3689','3690','3691']
        if output_names[i] in tensor_names:
            mace_output_file = "/root/workspace/tensors/mace_" + output_names[i]
            onnx_output_file = "/root/workspace/tensors/onnxruntime_" + output_names[i]
            np.savetxt(mace_output_file, mace_out_value, '%.6f')
            np.savetxt(onnx_output_file, value, '%.6f')
        compare_output(output_names[i],
                       mace_out_value, value,
                       validation_threshold, log_file,
                       real_output_shape, real_output_data_format)


def validate_megengine_model(platform, model_file, input_file,
                             mace_out_file, input_names, input_shapes,
                             input_data_formats, output_names, output_shapes,
                             output_data_formats, validation_threshold,
                             input_data_types, log_file):
    import megengine._internal as mgb

    if not os.path.isfile(model_file):
        common.MaceLogger.error(
            VALIDATION_MODULE,
            "Input graph file '" + model_file + "' does not exist!",
            )

    feed_inputs = []
    for i in range(len(input_names)):
        input_value = load_data(
            util.formatted_file_name(input_file, input_names[i]),
            input_data_types[i])
        input_value = input_value.reshape(input_shapes[i])
        if (input_data_formats[i] == DataFormat.NHWC and
                len(input_shapes[i]) == 4):
            input_value = input_value.transpose((0, 3, 1, 2))
        feed_inputs.append(input_value)

    cg, _, outputs = mgb.load_comp_graph_from_file(model_file)
    inputs = mgb.cgtools.get_dep_vars(outputs, "Host2DeviceCopy")
    inputs = sorted(inputs, key=lambda i: i.name)
    outputs = list(map(mgb.copy_output, outputs))
    if len(outputs) == 1:
        (outputs,) = outputs
    func = cg.compile(inputs, outputs)

    mge_output_value = func(*feed_inputs)

    for i in range(len(output_names)):
        output_file_name = \
            util.formatted_file_name(mace_out_file, output_names[i])
        mace_out_value = load_data(output_file_name)
        mace_out_value, real_output_shape, real_output_data_format = \
            get_real_out_value_shape_df(platform,
                                        mace_out_value,
                                        output_shapes[i],
                                        output_data_formats[i])
        compare_output(output_names[i], mace_out_value,
                       mge_output_value, validation_threshold, log_file,
                       real_output_shape, real_output_data_format)


def validate_keras_model(platform, model_file,
                         input_file, mace_out_file,
                         input_names, input_shapes, input_data_formats,
                         output_names, output_shapes, output_data_formats,
                         validation_threshold, input_data_types, log_file):
    from tensorflow import keras
    import tensorflow_model_optimization as tfmot

    if not os.path.isfile(model_file):
        util.MaceLogger.error(
            VALIDATION_MODULE,
            "Input model file '" + model_file + "' does not exist!")

    with tfmot.quantization.keras.quantize_scope():
        keras_model = keras.models.load_model(model_file, compile=False)

        input = []
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
            input.append(input_value)

        output_values = keras_model.predict(input)

        mace_check(len(output_names) == 1, "Unexpected")

        for i in range(len(output_names)):
            output_file_name = util.formatted_file_name(
                mace_out_file, output_names[i])
            mace_out_value = load_data(
                output_file_name,
                get_data_type_by_value(output_values))
            mace_out_value, real_output_shape, real_output_data_format = \
                get_real_out_value_shape_df(platform,
                                            mace_out_value,
                                            output_shapes[i],
                                            output_data_formats[i])
            compare_output(output_names[i],
                           mace_out_value, output_values,
                           validation_threshold, log_file,
                           real_output_shape, real_output_data_format)


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
        validate_with_file(platform, output_node, output_shape,
                           mace_out_file, validation_outputs,
                           validation_threshold, log_file,
                           output_data_format)
    elif platform == Platform.TENSORFLOW:
        validate_tf_model(platform, model_file, input_file, mace_out_file,
                          input_node, input_shape, input_data_format,
                          output_node, output_shape, output_data_format,
                          validation_threshold, input_data_type,
                          log_file)
    elif platform == Platform.PYTORCH:
        validate_pytorch_model(platform, model_file, input_file, mace_out_file,
                               input_node, input_shape, input_data_format,
                               output_node, output_shape, output_data_format,
                               validation_threshold, input_data_type,
                               log_file)
    elif platform == Platform.CAFFE:
        validate_caffe_model(platform, model_file,
                             input_file, mace_out_file, weight_file,
                             input_node, input_shape, input_data_format,
                             output_node, output_shape, output_data_format,
                             validation_threshold, log_file)
    elif platform == Platform.ONNX:
        validate_onnx_model(platform, model_file,
                            input_file, mace_out_file,
                            input_node, input_shape, input_data_format,
                            output_node, output_shape, output_data_format,
                            validation_threshold,
                            input_data_type, backend, log_file)
    elif platform == Platform.MEGENGINE:
        validate_megengine_model(platform, model_file,
                                 input_file, mace_out_file,
                                 input_node, input_shape,
                                 input_data_format,
                                 output_node, output_shape,
                                 output_data_format,
                                 validation_threshold,
                                 input_data_type, log_file)
    elif platform == Platform.KERAS:
        validate_keras_model(platform, model_file, input_file, mace_out_file,
                             input_node, input_shape, input_data_format,
                             output_node, output_shape, output_data_format,
                             validation_threshold, input_data_type,
                             log_file)
    else:
        mace_check(False, "Unsupported platform")
