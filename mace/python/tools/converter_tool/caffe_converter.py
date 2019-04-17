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


import math

import numpy as np
import six
import google.protobuf.text_format

from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter
from mace.python.tools.converter_tool import shape_inference
from mace.python.tools.converter_tool.base_converter import PoolingType
from mace.python.tools.converter_tool.base_converter import ActivationType
from mace.python.tools.converter_tool.base_converter import EltwiseType
from mace.python.tools.converter_tool.base_converter import FrameworkType
from mace.python.tools.converter_tool.base_converter import DataFormat
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.convert_util import mace_check

from third_party.caffe import caffe_pb2

caffe_group_str = 'group'
caffe_kernel_h_str = 'kernel_h'
caffe_kernel_w_str = 'kernel_w'
caffe_stride_h_str = 'stride_h'
caffe_stride_w_str = 'stride_w'
caffe_pad_h_str = 'pad_h'
caffe_pad_w_str = 'pad_w'


class CaffeOperator(object):
    """CaffeOperator merges and provides both layer and weights information.
    Layer records caffe layer proto, while blobs records the weight data in
    format of numpy ndarray.
    """

    def __init__(self):
        self._layer = None
        self._blobs = None

    @property
    def name(self):
        return self._layer.name

    @property
    def type(self):
        return self._layer.type

    @property
    def layer(self):
        return self._layer

    @property
    def blobs(self):
        return self._blobs

    @layer.setter
    def layer(self, layer):
        self._layer = layer

    @blobs.setter
    def blobs(self, blobs):
        self._blobs = [self.blob_to_nparray(blob) for blob in blobs]

    def get_blob(self, index):
        mace_check(index < len(self._blobs), "blob out of index")
        return self._blobs[index]

    @staticmethod
    def blob_to_nparray(blob):
        if blob.num != 0:
            return (np.asarray(blob.data, dtype=np.float32).reshape(
                (blob.num, blob.channels, blob.height, blob.width)))
        else:
            return np.asarray(blob.data, dtype=np.float32).reshape(
                blob.shape.dim)


class CaffeNet(object):
    """CaffeNet contains caffe operations. Output of each layer has unique
    name as we replace duplicated output name with unique one, while keep
    mace input/output name which user specifies unchanged."""

    def __init__(self):
        self._ops = {}
        self._consumers = {}
        # for in-place op, its input name is the same with output name,
        # so we change the output name to an alias
        self._alias_op_output_name = {}
        self._used_op_output_name = set()

    @property
    def ops(self):
        return self._ops.values()

    def get_op(self, op_name):
        return self._ops.get(op_name, None)

    def get_consumers(self, tensor_name):
        return self._consumers.get(tensor_name, [])

    def add_layer(self, layer):
        op = CaffeOperator()
        op.layer = layer
        self._ops[layer.name] = op

        # change op output name if it is an in-place op
        layer.bottom[:] = [self._alias_op_output_name.get(layer_input,
                                                          layer_input) for
                           layer_input in layer.bottom][:]
        for i in six.moves.range(len(layer.top)):
            old_name = layer.top[i]
            if layer.type == 'Input':
                new_name = old_name
            else:
                idx = 0
                new_name = old_name + '#' + str(idx)
                while new_name in self._used_op_output_name:
                    idx += 1
                    new_name = old_name + '#' + str(idx)
            layer.top[i] = new_name
            self._alias_op_output_name[old_name] = new_name
            self._used_op_output_name.update([new_name])
        for input_tensor in layer.bottom:
            if input_tensor not in self._consumers:
                self._consumers[input_tensor] = []
            self._consumers[input_tensor].append(op)

    def add_blob(self, weight):
        if weight.name in self._ops:
            op = self._ops[weight.name]
            op.blobs = list(weight.blobs)


class CaffeConverter(base_converter.ConverterInterface):
    """A class for convert caffe model to mace model."""

    pooling_type_mode = {
        caffe_pb2.PoolingParameter.AVE: PoolingType.AVG,
        caffe_pb2.PoolingParameter.MAX: PoolingType.MAX
    }
    eltwise_type = {
        caffe_pb2.EltwiseParameter.PROD: EltwiseType.PROD,
        caffe_pb2.EltwiseParameter.SUM: EltwiseType.SUM,
        caffe_pb2.EltwiseParameter.MAX: EltwiseType.MAX,
    }
    activation_type = {
        'ReLU': ActivationType.RELU,
        'PReLU': ActivationType.PRELU,
        'TanH': ActivationType.TANH,
        'Sigmoid': ActivationType.SIGMOID,
    }

    def __init__(self, option, src_model_file, src_weight_file):
        self._op_converters = {
            'Input': self.convert_nop,
            'Convolution': self.convert_conv2d,
            'Deconvolution': self.convert_deconv2d,
            'Eltwise': self.convert_elementwise,
            'Add': self.convert_add,
            'ReLU': self.convert_activation,
            'TanH': self.convert_activation,
            'Sigmoid': self.convert_activation,
            'PReLU': self.convert_activation,
            'Pooling': self.convert_pooling,
            'Concat': self.convert_concat,
            'Slice': self.convert_slice,
            'Softmax': self.convert_softmax,
            'InnerProduct': self.convert_fully_connected,
            'Interp': self.convert_interp,
            'BatchNorm': self.convert_folded_batchnorm,
            'Crop': self.convert_crop,
            'Scale': self.convert_scale,
            'ShuffleChannel': self.convert_channel_shuffle,
            'Permute': self.convert_permute,
            'Flatten': self.convert_flatten,
            'PriorBox': self.convert_prior_box,
            'Reshape': self.convert_reshape,
        }
        self._option = option
        self._mace_net_def = mace_pb2.NetDef()
        ConverterUtil.set_filter_format(self._mace_net_def, DataFormat.OIHW)
        ConverterUtil.add_data_format_arg(self._mace_net_def, DataFormat.NCHW)
        self._caffe_net = CaffeNet()
        self._caffe_layers = caffe_pb2.NetParameter()
        caffe_weights = caffe_pb2.NetParameter()

        # parse prototxt
        with open(src_model_file, 'r') as f:
            google.protobuf.text_format.Merge(
                str(f.read()), self._caffe_layers)
            self.filter_test_layers(self._caffe_layers)
            for layer in self._caffe_layers.layer:
                self._caffe_net.add_layer(layer)

        # parse model weight
        with open(src_weight_file, 'rb') as f:
            caffe_weights.ParseFromString(f.read())
            self.filter_test_layers(caffe_weights)
            for weight in caffe_weights.layer:
                self._caffe_net.add_blob(weight)

        self._skip_ops = []

    def run(self):
        self.convert_ops()
        shape_inferer = shape_inference.ShapeInference(
            self._mace_net_def,
            self._option.input_nodes.values())
        shape_inferer.run()
        self.replace_output_tensor_name()
        return self._mace_net_def

    @staticmethod
    def replace_input_name(ops, src_name, dst_name):
        for op in ops:
            for i in six.moves.range(len(op.input)):
                if op.input[i] == src_name:
                    op.input[i] = dst_name

    def replace_output_tensor_name(self):
        consumers = {}
        for op in self._mace_net_def.op:
            for input_name in op.input:
                if input_name not in consumers:
                    consumers[input_name] = []
                consumers[input_name].append(op)

        # replace the last op with same prefix name with the original top name
        ops = [op for op in self._mace_net_def.op]
        ops.reverse()
        visited = set()
        for op in ops:
            for i in six.moves.range(len(op.output)):
                original_output_name = op.output[i].split('#')[0]
                if original_output_name not in visited and\
                        original_output_name not in self._option.input_nodes:
                    self.replace_input_name(
                        consumers.get(op.output[i], []),
                        op.output[i],
                        original_output_name)
                    op.output[i] = original_output_name
                    visited.update([original_output_name])

        # if user set op name as output node, replace it with op name
        for op in self._mace_net_def.op:
            if op.name in self._option.output_nodes and op.name not in visited:
                if len(op.output) > 0:
                    self.replace_input_name(
                        consumers.get(op.output[0], []),
                        op.output,
                        op.name)
                    op.output[0] = op.name

    @staticmethod
    def filter_test_layers(layers):
        phase_map = {0: 'train', 1: 'test'}
        while True:
            changed = False
            for layer in layers.layer:
                phase = 'test'
                if len(layer.include):
                    phase = phase_map[layer.include[0].phase]
                if len(layer.exclude):
                    phase = phase_map[layer.exclude[0].phase]
                if phase != 'test' or layer.type == 'Dropout':
                    print("Remove layer %s (%s)" % (layer.name, layer.type))
                    layers.layer.remove(layer)
                    changed = True
                    break
            if not changed:
                break

    @staticmethod
    def add_stride_pad_kernel_arg(param, op_def):
        try:
            if len(param.stride) > 1 or len(param.kernel_size) > 1 or len(
                    param.pad) > 1:
                raise Exception(
                    'Mace does not support multiple stride/kernel_size/pad')
            stride = [param.stride[0],
                      param.stride[0]] if len(param.stride) else [1, 1]
            pad = [param.pad[0] * 2,
                   param.pad[0] * 2] if len(param.pad) else [0, 0]
            kernel = [param.kernel_size[0], param.kernel_size[0]] if len(
                param.kernel_size) else [0, 0]
        except TypeError:
            stride = [param.stride, param.stride]
            pad = [param.pad * 2, param.pad * 2]
            kernel = [param.kernel_size, param.kernel_size]

        if param.HasField(caffe_stride_h_str) or param.HasField(
                caffe_stride_w_str):
            stride = [param.stride_h, param.stride_w]
        if param.HasField(caffe_pad_h_str) or param.HasField(caffe_pad_w_str):
            pad = [param.pad_h * 2, param.pad_w * 2]

        strides_arg = op_def.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(stride)
        padding_arg = op_def.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_values_str
        padding_arg.ints.extend(pad)

        if op_def.type == MaceOp.Pooling.name:
            if param.HasField(caffe_kernel_h_str) or param.HasField(
                    caffe_kernel_w_str):
                kernel = [param.kernel_h, param.kernel_w]
            kernels_arg = op_def.arg.add()
            kernels_arg.name = MaceKeyword.mace_kernel_str
            kernels_arg.ints.extend(kernel)
            if param.HasField('global_pooling'):
                global_pooling_arg = op_def.arg.add()
                global_pooling_arg.name = MaceKeyword.mace_global_pooling_str
                global_pooling_arg.i = 1

    def convert_ops(self):
        layer_names = set()
        for layer in self._caffe_layers.layer:
            caffe_op = self._caffe_net.get_op(layer.name)
            if caffe_op not in self._skip_ops:
                mace_check(layer.name not in layer_names,
                           "There is duplicate layer name '%s' in your model"
                           % layer.name)
                mace_check(layer.type in self._op_converters,
                           "Mace does not support caffe op type %s yet"
                           % layer.type)
                layer_names.add(layer.name)
                self._op_converters[layer.type](caffe_op)

    def add_tensor(self, name, shape, data_type, value):
        tensor = self._mace_net_def.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_type = data_type
        tensor.float_data.extend(value.flat)

    def convert_nop(self, layer):
        pass

    def convert_general_op(self, caffe_op):
        op = self._mace_net_def.op.add()
        op.name = caffe_op.name
        op.type = caffe_op.type
        op.input.extend(caffe_op.layer.bottom)
        op.output.extend(caffe_op.layer.top)

        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        data_type_arg.i = self._option.data_type

        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.CAFFE.value

        ConverterUtil.add_data_format_arg(op, DataFormat.NCHW)

        return op

    def convert_conv2d(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.convolution_param
        is_depthwise = False
        if param.HasField(caffe_group_str) and param.group > 1:
            filter_data = caffe_op.blobs[0]
            mace_check(param.group == filter_data.shape[0] and
                       filter_data.shape[1] == 1,
                       "Mace do not support group convolution yet")
            is_depthwise = True
            caffe_op.blobs[0] = filter_data.reshape(1,
                                                    filter_data.shape[0],
                                                    filter_data.shape[2],
                                                    filter_data.shape[3])

        if is_depthwise:
            op.type = MaceOp.DepthwiseConv2d.name
        else:
            op.type = MaceOp.Conv2D.name

        self.add_stride_pad_kernel_arg(param, op)
        # dilation is specific for convolution in caffe
        dilations = [1, 1]
        if len(param.dilation) > 0:
            dilation_arg = op.arg.add()
            dilation_arg.name = MaceKeyword.mace_dilations_str
            if len(param.dilation) == 1:
                dilations = [param.dilation[0], param.dilation[0]]
            elif len(param.dilation) == 2:
                dilations = [param.dilation[0], param.dilation[1]]
            dilation_arg.ints.extend(dilations)

        filter_tensor_name = op.name + '_filter'
        filter_data = caffe_op.blobs[0]
        self.add_tensor(filter_tensor_name, filter_data.shape,
                        mace_pb2.DT_FLOAT, filter_data)
        op.input.extend([filter_tensor_name])

        if len(caffe_op.blobs) == 2:
            bias_tensor_name = op.name + '_bias'
            bias_data = caffe_op.blobs[1]
            # caffe of old version has 4-dimension bias, so reshape it
            # to single dimension
            self.add_tensor(bias_tensor_name, bias_data.reshape(-1).shape,
                            mace_pb2.DT_FLOAT,
                            bias_data)
            op.input.extend([bias_tensor_name])

    def convert_deconv2d(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.convolution_param

        if param.HasField(caffe_group_str) and param.group > 1:
            group_arg = op.arg.add()
            group_arg.name = MaceKeyword.mace_group_str
            group_arg.i = param.group
            op.type = MaceOp.DepthwiseDeconv2d.name
        else:
            op.type = MaceOp.Deconv2D.name

        self.add_stride_pad_kernel_arg(param, op)
        # dilation is specific for convolution in caffe
        dilations = [1, 1]
        if len(param.dilation) > 0:
            dilation_arg = op.arg.add()
            dilation_arg.name = MaceKeyword.mace_dilations_str
            if len(param.dilation) == 1:
                dilations = [param.dilation[0], param.dilation[0]]
            elif len(param.dilation) == 2:
                dilations = [param.dilation[0], param.dilation[1]]
            mace_check(dilations[0] == 1 and dilations[1] == 1,
                       "Mace only supports dilation == 1 deconvolution.")
            dilation_arg.ints.extend(dilations)

        filter_tensor_name = op.name + '_filter'
        filter_data = caffe_op.blobs[0]
        self.add_tensor(filter_tensor_name, filter_data.shape,
                        mace_pb2.DT_FLOAT, filter_data)
        op.input.extend([filter_tensor_name])

        if len(caffe_op.blobs) == 2:
            bias_tensor_name = op.name + '_bias'
            bias_data = caffe_op.blobs[1]
            # caffe of old version has 4-dimension bias, so reshape it
            # to single dimension
            self.add_tensor(bias_tensor_name, bias_data.reshape(-1).shape,
                            mace_pb2.DT_FLOAT,
                            bias_data)
            op.input.extend([bias_tensor_name])

    def convert_elementwise(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.eltwise_param

        op.type = MaceOp.Eltwise.name
        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        type_arg.i = self.eltwise_type[param.operation].value
        if len(param.coeff) > 0:
            coeff_arg = op.arg.add()
            coeff_arg.name = 'coeff'
            coeff_arg.floats.extend(list(param.coeff))

    def convert_add(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = MaceOp.AddN.name

    def convert_activation(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = MaceOp.Activation.name

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_activation_type_str
        type_arg.s = six.b(self.activation_type[caffe_op.type].name)

        if caffe_op.type == 'PReLU':
            alpha_tensor_name = caffe_op.name + '_alpha'
            alpha_data = caffe_op.blobs[0]
            self.add_tensor(alpha_tensor_name, alpha_data.reshape(-1).shape,
                            mace_pb2.DT_FLOAT, alpha_data)
            op.input.extend([alpha_tensor_name])

        negative_slope = caffe_op.layer.relu_param.negative_slope
        if caffe_op.type == 'ReLU' and negative_slope != 0:
            param_arg = op.arg.add()
            param_arg.name = MaceKeyword.mace_activation_leakyrelu_coefficient_str  # noqa
            param_arg.f = caffe_op.layer.relu_param.negative_slope

            type_arg.s = six.b(ActivationType.LEAKYRELU.name)

    def convert_folded_batchnorm(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = MaceOp.BatchNorm.name

        scale_op = None
        for consumer in self._caffe_net.get_consumers(caffe_op.layer.top[0]):
            if consumer.type == 'Scale':
                scale_op = consumer
        mace_check(scale_op is not None, "batchnorm is not followed by scale")
        self._skip_ops.append(scale_op)

        epsilon_value = caffe_op.layer.batch_norm_param.eps
        mace_check(caffe_op.blobs[2][0] != 0, "batchnorm scalar is zero")
        mean_value = (1. / caffe_op.blobs[2][0]) * caffe_op.blobs[0]
        var_value = (1. / caffe_op.blobs[2][0]) * caffe_op.blobs[1]
        gamma_value = scale_op.blobs[0]
        beta_value = np.zeros_like(mean_value)
        if len(scale_op.blobs) == 2:
            beta_value = scale_op.blobs[1]

        scale_value = (
                (1.0 / np.vectorize(math.sqrt)(var_value + epsilon_value)) *
                gamma_value).reshape(-1)
        offset_value = ((-mean_value * scale_value) + beta_value).reshape(-1)

        input_names = [op.name + '_scale', op.name + '_offset']
        self.add_tensor(input_names[0], scale_value.reshape(-1).shape,
                        mace_pb2.DT_FLOAT, scale_value)
        self.add_tensor(input_names[1], offset_value.reshape(-1).shape,
                        mace_pb2.DT_FLOAT, offset_value)
        op.input.extend([name for name in input_names])
        op.output[:] = scale_op.layer.top[:]

    def convert_pooling(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.pooling_param

        op.type = MaceOp.Pooling.name
        self.add_stride_pad_kernel_arg(param, op)
        pooling_type_arg = op.arg.add()
        pooling_type_arg.name = MaceKeyword.mace_pooling_type_str
        pooling_type_arg.i = self.pooling_type_mode[param.pool].value

    def convert_softmax(self, caffe_op):
        self.convert_general_op(caffe_op)

    def convert_crop(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.crop_param
        op.type = MaceOp.Crop.name

        axis = param.axis
        axis = 4 + axis if axis < 0 else axis
        offset_value = -1 * np.ones(4, dtype=np.int32)
        offset_len = len(param.offset)
        if offset_len == 1:
            while axis < 4:
                offset_value[axis] = param.offset[0]
                axis += 1
        else:
            offset_value[axis:] = param.offset

        offset_arg = op.arg.add()
        offset_arg.name = MaceKeyword.mace_offset_str
        offset_arg.ints.extend(offset_value)

    def convert_concat(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.concat_param
        op.type = MaceOp.Concat.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 1
        if param.HasField(MaceKeyword.mace_axis_str):
            axis_arg.i = param.axis
        elif param.HasField('concat_dim'):
            axis_arg.i = param.concat_dim

    def convert_slice(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = MaceOp.Split.name

        if caffe_op.layer.HasField('slice_param'):
            param = caffe_op.layer.slice_param
            mace_check(not param.HasField('axis') or param.axis == 1
                       or param.axis == -3,
                       "Mace do not support slice with axis %d" % param.axis)
            mace_check(len(param.slice_point) == 0,
                       "Mace do not support slice with slice_point")
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 1

    def convert_interp(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.interp_param
        mace_check(param.HasField("height") and param.HasField("width"),
                   'Only support bilinear interp with height and width')
        op.type = MaceOp.ResizeBilinear.name

        size_arg = op.arg.add()
        size_arg.name = MaceKeyword.mace_resize_size_str
        size_value = np.array([param.height, param.width], dtype=np.int32)
        size_arg.ints.extend(size_value)

    def convert_fully_connected(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.inner_product_param
        op.type = MaceOp.FullyConnected.name

        mace_check((param.axis == 1 or param.axis == -3)
                   and not param.transpose,
                   "Do not support non-default axis and transpose")
        mace_check(caffe_op.blobs[0].ndim in [2, 4],
                   "Unexpected fc weigth ndim.")
        if caffe_op.blobs[0].ndim == 4:
            mace_check(list(caffe_op.blobs[0].shape[:2]) == [1, 1],
                       "Do not support 4D weight with shape [1, 1, *, *]")

        weight_tensor_name = op.name + '_weight'
        weight_data = caffe_op.blobs[0].reshape(param.num_output, -1)
        self.add_tensor(weight_tensor_name, weight_data.shape,
                        mace_pb2.DT_FLOAT,
                        weight_data)
        op.input.extend([weight_tensor_name])

        if len(caffe_op.blobs) == 2:
            bias_tensor_name = op.name + '_bias'
            bias_data = caffe_op.blobs[1]
            self.add_tensor(bias_tensor_name, bias_data.reshape(-1).shape,
                            mace_pb2.DT_FLOAT,
                            bias_data)
            op.input.extend([bias_tensor_name])

    def convert_scale(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        op.type = MaceOp.Eltwise.name

        scale_op_name = op.name
        op.name = scale_op_name + '_prod'

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        type_arg.i = EltwiseType.PROD.value

        scale_tensor_name = scale_op_name + '_scale'
        scale_data = caffe_op.blobs[0]
        self.add_tensor(scale_tensor_name, scale_data.shape,
                        mace_pb2.DT_FLOAT, scale_data)
        op.input.extend([scale_tensor_name])

        if len(caffe_op.blobs) == 2:
            bias_tensor_name = scale_op_name + '_offset'
            bias_data = caffe_op.blobs[1]
            # caffe of old version has 4-dimension bias, so reshape it
            # to single dimension
            self.add_tensor(bias_tensor_name, bias_data.reshape(-1).shape,
                            mace_pb2.DT_FLOAT,
                            bias_data)
            op.input.extend([bias_tensor_name])

            biasadd_op = self._mace_net_def.op.add()
            biasadd_op.name = scale_op_name + '_biasadd'
            biasadd_op.type = MaceOp.BiasAdd.name
            biasadd_op.output.extend(op.output)
            op.output[:] = [op.output[0] + '_prod_output']
            biasadd_op.input.extend(op.output)
            biasadd_op.input.extend([op.input[2]])

            biasadd_op.output_shape.extend(op.output_shape)

            del op.input[2]

            data_type_arg = biasadd_op.arg.add()
            data_type_arg.name = 'T'
            data_type_arg.i = self._option.data_type

            ConverterUtil.add_data_format_arg(biasadd_op,
                                              DataFormat.NCHW)

    def convert_channel_shuffle(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.shuffle_channel_param
        op.type = MaceOp.ChannelShuffle.name

        group_arg = op.arg.add()
        group_arg.name = MaceKeyword.mace_group_str
        group_arg.i = 1
        if param.HasField('group'):
            group_arg.i = param.group

    def convert_permute(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.permute_param
        op.type = MaceOp.Transpose.name

        dims_arg = op.arg.add()
        dims_arg.name = MaceKeyword.mace_dims_str
        dims_arg.ints.extend(list(param.order))

    def convert_flatten(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.flatten_param
        op.type = MaceOp.Reshape.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 1
        if param.HasField('axis'):
            axis_arg.i = param.axis
        axis_arg.i = 4 + axis_arg.i if axis_arg.i < 0 else axis_arg.i

        end_axis_arg = op.arg.add()
        end_axis_arg.name = MaceKeyword.mace_end_axis_str
        end_axis_arg.i = -1
        if param.HasField('end_axis'):
            end_axis_arg.i = param.end_axis

    def convert_prior_box(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.prior_box_param
        op.type = MaceOp.PriorBox.name

        min_size_arg = op.arg.add()
        min_size_arg.name = MaceKeyword.mace_min_size_str
        min_size_arg.floats.extend(list(param.min_size))
        max_size_arg = op.arg.add()
        max_size_arg.name = MaceKeyword.mace_max_size_str
        max_size_arg.floats.extend(list(param.max_size))
        flip_arg = op.arg.add()
        flip_arg.name = MaceKeyword.mace_flip_str
        flip_arg.i = 1
        if param.HasField('flip'):
            flip_arg.i = int(param.flip)
        aspect_ratio = [1.0]
        for i in param.aspect_ratio:
            already_exist = False
            for ar in aspect_ratio:
                if abs(i - ar) < 1e-6:
                    already_exist = True
                    break
            if not already_exist:
                aspect_ratio.append(i)
                if flip_arg.i:
                    aspect_ratio.append(1.0 / i)
        aspect_ratio_arg = op.arg.add()
        aspect_ratio_arg.name = MaceKeyword.mace_aspect_ratio_str
        aspect_ratio_arg.floats.extend(list(aspect_ratio))
        clip_arg = op.arg.add()
        clip_arg.name = MaceKeyword.mace_clip_str
        clip_arg.i = 0
        if param.HasField('clip'):
            clip_arg.i = int(param.clip)
        variance_arg = op.arg.add()
        variance_arg.name = MaceKeyword.mace_variance_str
        variance_arg.floats.extend(list(param.variance))
        offset_arg = op.arg.add()
        offset_arg.name = MaceKeyword.mace_offset_str
        offset_arg.f = 0.5
        if param.HasField('offset'):
            offset_arg.f = param.offset
        step_h_arg = op.arg.add()
        step_h_arg.name = MaceKeyword.mace_step_h_str
        step_h_arg.f = 0
        if param.HasField('step_h'):
            mace_check(not param.HasField('step'),
                       "Either step or step_h/step_w should be specified; not both.")  # noqa
            step_h_arg.f = param.step_h
            mace_check(step_h_arg.f > 0, "step_h should be larger than 0.")
        step_w_arg = op.arg.add()
        step_w_arg.name = MaceKeyword.mace_step_w_str
        step_w_arg.f = 0
        if param.HasField('step_w'):
            mace_check(not param.HasField('step'),
                       "Either step or step_h/step_w should be specified; not both.")  # noqa
            step_w_arg.f = param.step_w
            mace_check(step_w_arg.f > 0, "step_w should be larger than 0.")

        if param.HasField('step'):
            mace_check(not param.HasField('step_h') and not param.HasField('step_w'),  # noqa
                       "Either step or step_h/step_w should be specified; not both.")  # noqa
            mace_check(param.step > 0, "step should be larger than 0.")
            step_h_arg.f = param.step
            step_w_arg.f = param.step

    def convert_reshape(self, caffe_op):
        op = self.convert_general_op(caffe_op)
        param = caffe_op.layer.reshape_param
        op.type = MaceOp.Reshape.name

        dim_arg = op.arg.add()
        dim_arg.name = MaceKeyword.mace_dim_str
        dim_arg.ints.extend(list(param.shape.dim))

        axis_arg = op.arg.add()
        axis_arg.name = 'reshape_' + MaceKeyword.mace_axis_str
        axis_arg.i = 0
        if param.HasField('axis'):
            axis_arg.i = param.axis

        num_axes_arg = op.arg.add()
        num_axes_arg.name = MaceKeyword.mace_num_axes_str
        num_axes_arg.i = -1
        if param.HasField('num_axes'):
            num_axes_arg.i = param.num_axes
