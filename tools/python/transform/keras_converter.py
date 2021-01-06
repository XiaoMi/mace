import sys
import copy

from enum import Enum
import six

from py_proto import mace_pb2
from transform import base_converter
from transform.base_converter import ActivationType
from transform.base_converter import ConverterUtil
from transform.base_converter import DataFormat
from transform.base_converter import EltwiseType
from transform.base_converter import FrameworkType
from transform.base_converter import MaceOp
from transform.base_converter import MaceKeyword
from transform.base_converter import PoolingType
from transform.base_converter import PaddingMode
from transform.base_converter import PadType
from transform.base_converter import ReduceType
from transform.base_converter import RoundMode
from tensorflow import keras
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras import activations

from quantize import quantize_util
from utils.util import mace_check

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.\
    quantization.keras.quantize_layer import QuantizeLayer
from tensorflow_model_optimization.python.core.\
    quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow_model_optimization.python.core.\
    quantization.keras.quantize_annotate import QuantizeAnnotate

import numpy as np

padding_mode = {
    "valid": PaddingMode.VALID,
    "same": PaddingMode.SAME
    # 'full': PaddingMode.FULL
}


def dtype2mtype(dtype):
    if dtype == "float32":
        return mace_pb2.DT_FLOAT
    if dtype == "int32":
        return mace_pb2.DT_INT32
    if dtype == "int8":
        return mace_pb2.INT8

    mace_check(False, "data type %s not supported" % dtype)
    return None


def keras_shape2list(shape):
    dims = shape.as_list()
    for i in range(len(dims)):
        if dims[i] is None:
            dims[i] = 1

    return dims


def get_input(keras_op):
    if hasattr(keras_op, "input_proxy"):
        return keras_op.input_proxy
    else:
        return keras_op.input


def get_output(keras_op):
    if hasattr(keras_op, "output_proxy"):
        return keras_op.output_proxy
    else:
        return keras_op.output


def get_output_max(keras_op):
    for weight in keras_op.weights:
        last_name = weight.name.split('/')[-1].split(':')[0]
        if (last_name in
                ["post_activation_max", "output_max", "pre_activation_max"]):
            return weight.numpy()

    mace_check(False, "No output_max info in %s" % keras_op.name)


def get_output_min(keras_op):
    for weight in keras_op.weights:
        last_name = weight.name.split('/')[-1].split(':')[0]
        if (last_name in
                ["post_activation_min", "output_min", "pre_activation_min"]):
            return weight.numpy()

    mace_check(False, "No output_min info in %s" % keras_op.name)


def get_quant_wrapper_kernel(quant_wrapper):
    for weight in quant_wrapper.weights:
        if weight.name == quant_wrapper.name + "/kernel:0":
            return weight

    return None


def get_quant_wrapper_bias(quant_wrapper):
    for weight in quant_wrapper.weights:
        if weight.name == quant_wrapper.name + "/bias:0":
            return weight

    return None


def get_quant_wrapper_depthwise_kernel(quant_wrapper):
    for weight in quant_wrapper.weights:
        if weight.name == quant_wrapper.name + "/depthwise_kernel:0":
            return weight

    return None


def conv_output_length(input_length,
                       filter_size,
                       padding,
                       stride,
                       dilation=1):
    if input_length is None:
        return None

    mace_check(padding in {'same', 'valid', 'full', 'causal'},
               "Not supported padding type: %s" % padding)
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ['same', 'causal']:
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1)  # stride


activation_types_dict = {
    "relu": ActivationType.RELU,
    # 'relu6': ActivationType.RELUX,
    # 'PReLU': ActivationType.PRELU,
    # 'TanH': ActivationType.TANH,
    "sigmoid": ActivationType.SIGMOID
    # 'Clip': ActivationType.RELUX,
}


class KerasConverter(base_converter.ConverterInterface):
    """A class for convert tensorflow 2.0 keras h5 model to mace model."""

    def __init__(self, option, src_model_file):
        self._op_converters = {
            keras.layers.InputLayer: self.convert_input_layer,
            keras.layers.Flatten: self.convert_flatten,
            keras.layers.Dense: self.convert_dense,
            keras.layers.Conv2D: self.convert_conv2d,
            keras.layers.MaxPooling2D: self.convert_maxpooling2d,
            keras.layers.Dropout: self.convert_dropout,
            keras.layers.DepthwiseConv2D: self.convert_depthwise_conv2d,
            keras.layers.Softmax: self.convert_softmax,
            keras.layers.BatchNormalization: self.convert_batch_normalization,
            keras.layers.SeparableConv2D: self.convert_separable_conv2d,
            keras.layers.UpSampling2D: self.convert_upsampling2d,
            keras.layers.Activation: self.convert_activation,
            keras.layers.ReLU: self.convert_relu,
            keras.layers.Concatenate: self.convert_concatenate,
            keras.layers.GlobalAveragePooling2D:
                self.convert_global_average_pooling2d,
            keras.layers.Add: self.convert_add,
            QuantizeLayer: self.convert_quantize_layer,
            QuantizeWrapper: self.convert_quantize_wrapper,
            # keras.Sequential: self.convert_sequential,
        }

        self._option = option
        self._mace_net_def = mace_pb2.NetDef()
        ConverterUtil.set_filter_format(self._mace_net_def, DataFormat.HWIO)
        ConverterUtil.add_data_format_arg(self._mace_net_def, DataFormat.NHWC)

        with tfmot.quantization.keras.quantize_scope():
            self._keras_model = keras.models.load_model(src_model_file,
                                                        compile=False)
            self._keras_model.summary()

    def run(self):
        for op in self._keras_model.layers:
            mace_check(
                type(op) in self._op_converters,
                "Mace does not support keras op type %s yet" % type(op))
            self._op_converters[type(op)](op)

        return self._mace_net_def

    def convert_general_op(self, keras_op):
        op = self._mace_net_def.op.add()
        op.name = keras_op.name
        data_type_arg = op.arg.add()
        data_type_arg.name = "T"
        data_type_arg.i = dtype2mtype(keras_op.dtype)
        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.KERAS.value
        ConverterUtil.add_data_format_arg(op, DataFormat.NHWC)

        return op

    def convert_general_op_with_input_output(self, keras_op):
        op = self._mace_net_def.op.add()
        op.name = keras_op.name
        data_type_arg = op.arg.add()
        data_type_arg.name = "T"
        data_type_arg.i = dtype2mtype(keras_op.dtype)
        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.KERAS.value
        ConverterUtil.add_data_format_arg(op, DataFormat.NHWC)

        input = get_input(keras_op)
        if isinstance(input, list):
            for e in input:
                op.input.append(e.name)
        else:
            op.input.append(input.name)

        output = get_output(keras_op)
        mace_check(not isinstance(output, list), "only support one output")
        op.output.append(output.name)
        output_shape = op.output_shape.add()
        output_shape.dims.extend(keras_shape2list(output.shape))

        return op

    def convert_input_layer(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        op.type = MaceOp.Identity.name

        return op

    def convert_flatten(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        op.type = MaceOp.Reshape.name

        dim_arg = op.arg.add()
        dim_arg.name = MaceKeyword.mace_dim_str
        dim_arg.ints.extend([0, -1])

        return op

    def convert_dense(self, keras_op):
        op = self.convert_general_op(keras_op)
        op.type = MaceOp.MatMul.name

        op.input.append(get_input(keras_op).name)

        # Adds kernel tensor
        op.input.append(keras_op.kernel.name)
        kernel = self.add_keras_tensor(keras_op.kernel)

        # Adds bias tensor
        if keras_op.use_bias:
            op.input.append(keras_op.bias.name)
            self.add_keras_tensor(keras_op.bias)

        act_op = self.split_activation_op(keras_op, op)
        return [op, act_op]

    def convert_conv2d(self, keras_op):
        op = self.convert_general_op(keras_op)
        op.type = MaceOp.Conv2D.name
        op.input.append(get_input(keras_op).name)

        # Adds kernel tensor
        op.input.append(keras_op.kernel.name)
        kernel = self.add_keras_tensor(keras_op.kernel)

        # Adds bias tensor
        if keras_op.use_bias:
            op.input.append(keras_op.bias.name)
            self.add_keras_tensor(keras_op.bias)

        padding_arg = op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_str
        padding_arg.i = padding_mode[keras_op.padding].value

        strides_arg = op.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(keras_op.strides)

        dilation_arg = op.arg.add()
        dilation_arg.name = MaceKeyword.mace_dilations_str
        dilation_arg.ints.extend(keras_op.dilation_rate)

        act_op = self.split_activation_op(keras_op, op)
        return [op, act_op]

    def convert_depthwise_conv2d(self, keras_op):
        op = self.convert_general_op(keras_op)
        op.type = MaceOp.DepthwiseConv2d.name
        op.input.append(get_input(keras_op).name)

        # Adds kernel tensor
        op.input.append(keras_op.depthwise_kernel.name)
        kernel = self.add_keras_tensor(keras_op.depthwise_kernel)

        # Adds bias tensor
        if keras_op.use_bias:
            op.input.append(keras_op.bias.name)
            self.add_keras_tensor(keras_op.bias)

        padding_arg = op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_str
        padding_arg.i = padding_mode[keras_op.padding].value

        strides_arg = op.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(keras_op.strides)

        dilation_arg = op.arg.add()
        dilation_arg.name = MaceKeyword.mace_dilations_str
        dilation_arg.ints.extend(keras_op.dilation_rate)

        act_op = self.split_activation_op(keras_op, op)
        return [op, act_op]

    def convert_maxpooling2d(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        op.type = MaceOp.Pooling.name

        pooling_type_arg = op.arg.add()
        pooling_type_arg.name = MaceKeyword.mace_pooling_type_str
        pooling_type_arg.i = PoolingType.MAX.value

        padding_arg = op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_str
        padding_arg.i = padding_mode[keras_op.padding].value

        strides_arg = op.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(keras_op.strides)

        kernels_arg = op.arg.add()
        kernels_arg.name = MaceKeyword.mace_kernel_str
        kernels_arg.ints.extend(keras_op.pool_size)

        return op

    def convert_softmax(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        op.type = MaceOp.Softmax.name

        return op

    def convert_dropout(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        op.type = MaceOp.Identity.name

        return op

    def convert_batch_normalization(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        op.type = MaceOp.BatchNorm.name
        gamma = keras_op.gamma.numpy()
        beta = keras_op.beta.numpy()
        mean = keras_op.moving_mean.numpy()
        variance = keras_op.moving_variance.numpy()
        epsilon = keras_op.epsilon
        scale = (1.0 / np.sqrt(variance + epsilon)) * gamma
        offset = (-mean * scale) + beta
        scale_name = keras_op.name + '/scale:0'
        offset_name = keras_op.name + '/offset:0'
        self.add_numpy_tensor(scale_name, scale)
        self.add_numpy_tensor(offset_name, offset)
        op.input.extend([scale_name, offset_name])

        return op

    def convert_global_average_pooling2d(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        op.type = MaceOp.Reduce.name

        reduce_type_arg = op.arg.add()
        reduce_type_arg.name = MaceKeyword.mace_reduce_type_str
        reduce_type_arg.i = ReduceType.MEAN.value

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.ints.extend([1, 2])
        keep_dims_arg = op.arg.add()
        keep_dims_arg.name = MaceKeyword.mace_keepdims_str
        keep_dims_arg.i = 1

        origin_output_shape = copy.deepcopy(op.output_shape[0].dims)
        op.output_shape[0].dims.insert(1, 1)
        op.output_shape[0].dims.insert(1, 1)

        output_name = op.output[0]
        del op.output[:]
        output_name_mid = output_name + "_mid_reshape"
        op.output.append(output_name_mid)
        op_reshape = self._mace_net_def.op.add()
        op_reshape.name = keras_op.name + "_reshape"
        op_reshape.type = MaceOp.Reshape.name
        op_reshape.input.append(output_name_mid)
        op_reshape.output.append(output_name)
        output_shape = op_reshape.output_shape.add()
        output_shape.dims.extend(origin_output_shape)

        t_shape = list(origin_output_shape)
        shape_tensor_name = op_reshape.name + "_dest_shape"
        self.add_tensor(
            shape_tensor_name, [len(t_shape)], mace_pb2.DT_INT32, t_shape
        )
        op_reshape.input.append(shape_tensor_name)

        data_type_arg = op_reshape.arg.add()
        data_type_arg.name = "T"
        data_type_arg.i = dtype2mtype(keras_op.dtype)
        framework_type_arg = op_reshape.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.KERAS.value
        ConverterUtil.add_data_format_arg(op_reshape, DataFormat.NHWC)

        return op_reshape

    def convert_activation(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        activation = keras_op.activation

        if activation == activations.linear:
            op.type = MaceOp.Identity.name
        elif activation is activations.relu:
            op.type = MaceOp.Activation.name
            type_arg = op.arg.add()
            type_arg.name = MaceKeyword.mace_activation_type_str
            type_arg.s = six.b("RELU")
        elif activation == activations.softmax:
            op.type = MaceOp.Softmax.name
        else:
            mace_check(False, "Unsupported activation")

        return op

    def convert_relu(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        op.type = MaceOp.Activation.name
        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_activation_type_str
        type_arg.s = six.b("RELU")

        return op

    def convert_concatenate(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        op.type = MaceOp.Concat.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis = keras_op.axis
        axis = len(op.output_shape[0].dims) + axis if axis < 0 else axis
        axis_arg.i = axis
        return op

    def convert_add(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)
        op.type = MaceOp.Eltwise.name

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        type_arg.i = EltwiseType.SUM.value

        return op

    def convert_quantize_layer(self, keras_op):
        op = self._mace_net_def.op.add()
        op.name = keras_op.name
        op.type = MaceOp.Identity.name
        op.input.append(get_input(keras_op).name)
        op.output.append(get_output(keras_op).name)
        output_shape = op.output_shape.add()
        output_shape.dims.extend(keras_shape2list(get_output(keras_op).shape))

        ConverterUtil.add_data_type_arg(op, mace_pb2.DT_FLOAT)
        ConverterUtil.add_data_format_arg(op, DataFormat.NHWC)

        output_min = keras_op.weights[0].numpy()
        output_max = keras_op.weights[1].numpy()

        self.add_quantize_info(op, output_min, output_max)

        return op

    def convert_quantize_wrapper(self, keras_op_wrapper):
        inside_layer = keras_op_wrapper.layer
        if isinstance(inside_layer, convolutional.DepthwiseConv2D):
            inside_layer.depthwise_kernel = \
                get_quant_wrapper_depthwise_kernel(keras_op_wrapper)
            inside_layer.bias = get_quant_wrapper_bias(keras_op_wrapper)
        elif isinstance(inside_layer, convolutional.Conv):
            inside_layer.kernel = get_quant_wrapper_kernel(keras_op_wrapper)
            inside_layer.bias = get_quant_wrapper_bias(keras_op_wrapper)
        elif isinstance(inside_layer, keras.layers.Dense):
            inside_layer.kernel = get_quant_wrapper_kernel(keras_op_wrapper)
            inside_layer.bias = get_quant_wrapper_bias(keras_op_wrapper)

        # Adds input name for inside layers
        inside_layer.input_proxy = keras_op_wrapper.input
        inside_layer.output_proxy = keras_op_wrapper.output

        op = self._op_converters[type(inside_layer)](inside_layer)

        if isinstance(
                inside_layer,
                (convolutional.Conv, keras.layers.Dense,
                 keras.layers.Activation)):
            output_min = get_output_min(keras_op_wrapper)
            output_max = get_output_max(keras_op_wrapper)

            if not isinstance(op, list):
                self.add_quantize_info(op, output_min, output_max)
            else:
                assert len(op) == 2

                if op[1] is None:
                    self.add_quantize_info(op[0], output_min, output_max)
                else:
                    if op[1].type == MaceOp.Softmax.name:
                        self.add_quantize_info(op[0], output_min, output_max)
                    else:
                        self.add_quantize_info(op[1], output_min, output_max)

        return op

    def add_keras_tensor(self, keras_tensor):
        tensor = self._mace_net_def.tensors.add()
        tensor.name = keras_tensor.name
        tensor.dims.extend(keras_tensor.shape)
        tensor.data_type = dtype2mtype(keras_tensor.dtype)
        tensor.float_data.extend(keras_tensor.numpy().flat)
        return tensor

    def add_numpy_tensor(self, name, np_tensor):
        tensor = self._mace_net_def.tensors.add()
        tensor.name = name
        tensor.dims.extend(np_tensor.shape)
        tensor.data_type = dtype2mtype(np_tensor.dtype)
        tensor.float_data.extend(np_tensor.flat)
        return tensor

    def add_tensor(self, name, shape, data_type, value):
        tensor = self._mace_net_def.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_type = data_type
        if data_type == mace_pb2.DT_INT32:
            tensor.int32_data.extend(value)
        else:
            tensor.float_data.extend(value)

    def split_activation_op(self, keras_op, op):
        activation = keras_op.get_config()["activation"]
        if "class_name" in activation:
            assert activation["class_name"] == "QuantizeAwareActivation"
            activation = activation["config"]["activation"]

        if activation == "linear":
            op.output.append(get_output(keras_op).name)
            output_shape = op.output_shape.add()
            output_shape.dims.extend(
                keras_shape2list(get_output(keras_op).shape)
            )

            return None
        else:
            activation_tmp_name = get_output(keras_op).name + "_act"
            op.output.append(activation_tmp_name)
            output_shape = op.output_shape.add()
            output_shape.dims.extend(
                keras_shape2list(get_output(keras_op).shape)
            )

            activation_op = self._mace_net_def.op.add()
            activation_op.name = keras_op.name + "_act"
            if activation == "softmax":
                activation_op.type = MaceOp.Softmax.name
            else:
                activation_op.type = MaceOp.Activation.name
                type_arg = activation_op.arg.add()
                type_arg.name = MaceKeyword.mace_activation_type_str
                type_arg.s = six.b(activation_types_dict[activation].name)

            activation_op.input.append(activation_tmp_name)
            activation_op.output.append(get_output(keras_op).name)
            output_shape = activation_op.output_shape.add()
            output_shape.dims.extend(
                keras_shape2list(get_output(keras_op).shape)
            )

            data_type_arg = activation_op.arg.add()
            data_type_arg.name = "T"
            data_type_arg.i = dtype2mtype(keras_op.dtype)
            framework_type_arg = activation_op.arg.add()
            framework_type_arg.name = MaceKeyword.mace_framework_type_str
            framework_type_arg.i = FrameworkType.KERAS.value
            ConverterUtil.add_data_format_arg(activation_op, DataFormat.NHWC)

            return activation_op

    # Not supportted yet
    # def convert_sequential(self, engine):
    #     for op in engine.layers:
    #         mace_check(
    #             type(op) in self._op_converters,
    #             "Mace does not support keras op type %s yet" % type(op))
    #         self._op_converters[type(op)](op)

    def add_quantize_info(self, op, minval, maxval):
        quantize_schema = self._option.quantize_schema
        if quantize_schema == MaceKeyword.mace_apu_16bit_per_tensor:
            maxval = max(abs(minval), abs(maxval))
            minval = -maxval
            scale = maxval / 2 ** 15
            zero = 0
        elif quantize_schema == MaceKeyword.mace_int8:
            scale, zero, minval, maxval = quantize_util.adjust_range_int8(
                minval, maxval
            )
        else:
            scale, zero, minval, maxval = quantize_util.adjust_range(
                minval, maxval, self._option.device, non_zero=False
            )

        quantize_info = op.quantize_info.add()
        quantize_info.minval = minval
        quantize_info.maxval = maxval
        quantize_info.scale = scale
        quantize_info.zero_point = zero

        return quantize_info

    def convert_upsampling2d(self, keras_op):
        op = self.convert_general_op_with_input_output(keras_op)

        if keras_op.interpolation == 'nearest':
            op.type = MaceOp.ResizeNearestNeighbor.name
        else:
            op.type = MaceOp.ResizeBilinear.name

        height_scale_arg = op.arg.add()
        height_scale_arg.name = MaceKeyword.mace_height_scale_str
        width_scale_arg = op.arg.add()
        width_scale_arg.name = MaceKeyword.mace_width_scale_str
        height_scale_arg.f = keras_op.size[0]
        width_scale_arg.f = keras_op.size[1]

    def convert_separable_conv2d(self, keras_op):
        dw_conv2d_op = self.convert_general_op(keras_op)
        dw_conv2d_op.type = MaceOp.DepthwiseConv2d.name
        dw_conv2d_op.input.append(get_input(keras_op).name)
        # Adds kernel tensor
        dw_conv2d_op.input.append(keras_op.depthwise_kernel.name)
        dw_kernel = self.add_keras_tensor(keras_op.depthwise_kernel)

        padding_arg = dw_conv2d_op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_str
        padding_arg.i = padding_mode[keras_op.padding].value

        strides_arg = dw_conv2d_op.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(keras_op.strides)

        dilation_arg = dw_conv2d_op.arg.add()
        dilation_arg.name = MaceKeyword.mace_dilations_str
        dilations = keras_op.dilation_rate
        dilation_arg.ints.extend(keras_op.dilation_rate)

        dw_conv2d_output_name = keras_op.name + "_dw"
        dw_conv2d_op.output.append(dw_conv2d_output_name)

        input_shape = keras_shape2list(get_input(keras_op).shape)

        height = conv_output_length(input_shape[1],
                                    dw_kernel.dims[0],
                                    keras_op.padding,
                                    keras_op.strides[0],
                                    dilations[0])
        width = conv_output_length(input_shape[2],
                                   dw_kernel.dims[1],
                                   keras_op.padding,
                                   keras_op.strides[1],
                                   dilations[1])

        output_shape = dw_conv2d_op.output_shape.add()
        output_shape.dims.extend([input_shape[0],
                                  height,
                                  width,
                                  dw_kernel.dims[2] * dw_kernel.dims[3]])

        pw_conv2d_name = keras_op.name + "_pw"

        pw_conv2d_op = self._mace_net_def.op.add()
        pw_conv2d_op.name = pw_conv2d_name
        pw_conv2d_op.type = MaceOp.Conv2D.name

        pw_conv2d_op.input.append(dw_conv2d_output_name)

        # Adds kernel tensor
        pw_conv2d_op.input.append(keras_op.pointwise_kernel.name)
        self.add_keras_tensor(keras_op.pointwise_kernel)

        strides_arg = pw_conv2d_op.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend([1, 1])

        data_type_arg = pw_conv2d_op.arg.add()
        data_type_arg.name = "T"
        data_type_arg.i = dtype2mtype(keras_op.dtype)
        framework_type_arg = pw_conv2d_op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.KERAS.value
        ConverterUtil.add_data_format_arg(pw_conv2d_op, DataFormat.NHWC)

        # Adds bias tensor
        if keras_op.use_bias:
            pw_conv2d_op.input.append(keras_op.bias.name)
            self.add_keras_tensor(keras_op.bias)

        self.split_activation_op(keras_op, pw_conv2d_op)
