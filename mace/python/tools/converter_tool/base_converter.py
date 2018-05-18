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


from enum import Enum

from mace.proto import mace_pb2


class DataFormat(Enum):
    NHWC = 0
    NCHW = 1


class FilterFormat(Enum):
    HWIO = 0
    OIHW = 1
    HWOI = 2


class PaddingMode(Enum):
    VALID = 0
    SAME = 1
    FULL = 2


class PoolingType(Enum):
    AVG = 1
    MAX = 2


class ActivationType(Enum):
    NOOP = 0
    RELU = 1
    RELUX = 2
    PRELU = 3
    TANH = 4
    SIGMOID = 5


class EltwiseType(Enum):
    SUM = 0
    SUB = 1
    PROD = 2
    DIV = 3
    MIN = 4
    MAX = 5
    NEG = 6
    ABS = 7
    SQR_DIFF = 8
    POW = 9


MaceSupportedOps = [
    'Activation',
    'AddN',
    'BatchNorm',
    'BatchToSpaceND',
    'BiasAdd',
    'ChannelShuffle',
    'Concat',
    'Conv2D',
    'Deconv2D',
    'DepthToSpace',
    'DepthwiseConv2d',
    'Dequantize',
    'Eltwise',
    'FoldedBatchNorm',
    'FullyConnected',
    'LocalResponseNorm',
    'MatMul',
    'Pad',
    'Pooling',
    'Proposal',
    'PSROIAlign',
    'Quantize',
    'Requantize',
    'Reshape',
    'ResizeBilinear',
    'Slice',
    'Softmax',
    'SpaceToBatchND',
    'SpaceToDepth',
    'Transpose',
    'WinogradInverseTransform',
    'WinogradTransform',
]

MaceOp = Enum('MaceOp', [(op, op) for op in MaceSupportedOps], type=str)


class MaceKeyword(object):
    # node related str
    mace_input_node_name = 'mace_input_node'
    mace_output_node_name = 'mace_output_node'
    mace_buffer_type = 'buffer_type'
    mace_mode = 'mode'
    mace_buffer_to_image = 'BufferToImage'
    mace_image_to_buffer = 'ImageToBuffer'
    # arg related str
    mace_padding_str = 'padding'
    mace_padding_values_str = 'padding_values'
    mace_strides_str = 'strides'
    mace_dilations_str = 'dilations'
    mace_pooling_type_str = 'pooling_type'
    mace_global_pooling_str = 'global_pooling'
    mace_kernel_str = 'kernels'
    mace_data_format_str = 'data_format'
    mace_filter_format_str = 'filter_format'
    mace_element_type_str = 'type'
    mace_activation_type_str = 'activation'
    mace_activation_max_limit_str = 'max_limit'
    mace_resize_size_str = 'size'
    mace_batch_to_space_crops_str = 'crops'
    mace_paddings_str = 'paddings'
    mace_align_corners_str = 'align_corners'
    mace_space_batch_block_shape_str = 'block_shape'
    mace_space_depth_block_size_str = 'block_size'
    mace_constant_value_str = 'constant_value'
    mace_dims_str = 'dims'
    mace_axis_str = 'axis'
    mace_shape_str = 'shape'
    mace_winograd_filter_transformed = 'is_filter_transformed'
    mace_device = 'device'


class TransformerRule(Enum):
    REMOVE_USELESS_RESHAPE_OP = 0
    REMOVE_IDENTITY_OP = 1
    TRANSFORM_GLOBAL_POOLING = 2
    FOLD_RESHAPE = 3
    TRANSFORM_MATMUL_TO_FC = 4
    FOLD_BATCHNORM = 5
    FOLD_CONV_AND_BN = 6
    FOLD_DEPTHWISE_CONV_AND_BN = 7
    TRANSFORM_GPU_WINOGRAD = 8
    TRANSFORM_ADD_TO_BIASADD = 9
    FOLD_BIASADD = 10
    FOLD_ACTIVATION = 11
    TRANSPOSE_FILTERS = 12
    RESHAPE_FC_WEIGHT = 13
    TRANSPOSE_DATA_FORMAT = 14
    TRANSFORM_GLOBAL_CONV_TO_FC = 15
    TRANSFORM_BUFFER_IMAGE = 16
    ADD_DEVICE_AND_DATA_TYPE = 17
    SORT_BY_EXECUTION = 18


class ConverterInterface(object):
    """Base class for converting external models to mace models."""

    def run(self):
        raise NotImplementedError('run')


class NodeInfo(object):
    """A class for describing node information"""

    def __init__(self):
        self._name = None
        self._shape = []

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @name.setter
    def name(self, name):
        self._name = name

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    def __str__(self):
        return '%s %s' % (self._name, str(self._shape))


class ConverterOption(object):
    """A class for specifying options passed to converter tool"""

    def __init__(self):
        self._input_nodes = {}
        self._output_nodes = {}
        self._data_type = mace_pb2.DT_FLOAT
        self._device = mace_pb2.CPU
        self._winograd_enabled = False
        self._transformer_option = [
            TransformerRule.REMOVE_USELESS_RESHAPE_OP,
            TransformerRule.REMOVE_IDENTITY_OP,
            TransformerRule.TRANSFORM_GLOBAL_POOLING,
            TransformerRule.FOLD_RESHAPE,
            TransformerRule.TRANSFORM_MATMUL_TO_FC,
            TransformerRule.FOLD_BATCHNORM,
            TransformerRule.FOLD_CONV_AND_BN,
            TransformerRule.FOLD_DEPTHWISE_CONV_AND_BN,
            TransformerRule.TRANSFORM_GPU_WINOGRAD,
            TransformerRule.TRANSFORM_ADD_TO_BIASADD,
            TransformerRule.FOLD_BIASADD,
            TransformerRule.FOLD_ACTIVATION,
            TransformerRule.TRANSPOSE_FILTERS,
            TransformerRule.TRANSPOSE_DATA_FORMAT,
            TransformerRule.TRANSFORM_GLOBAL_CONV_TO_FC,
            TransformerRule.RESHAPE_FC_WEIGHT,
            TransformerRule.TRANSFORM_BUFFER_IMAGE,
            TransformerRule.ADD_DEVICE_AND_DATA_TYPE,
            TransformerRule.SORT_BY_EXECUTION,
        ]

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def output_nodes(self):
        return self._output_nodes

    @property
    def data_type(self):
        return self._data_type

    @property
    def device(self):
        return self._device

    @property
    def winograd_enabled(self):
        return self._winograd_enabled

    @property
    def transformer_option(self):
        return self._transformer_option

    @input_nodes.setter
    def input_nodes(self, input_nodes):
        for node in input_nodes:
            self._input_nodes[node.name] = node

    def add_input_node(self, input_node):
        self._input_nodes[input_node.name] = input_node

    @output_nodes.setter
    def output_nodes(self, output_nodes):
        for node in output_nodes:
            self.output_nodes[node.name] = node

    def add_output_node(self, output_node):
        self._output_nodes[output_node.name] = output_node

    @data_type.setter
    def data_type(self, data_type):
        self._data_type = data_type

    @device.setter
    def device(self, device):
        self._device = device

    @winograd_enabled.setter
    def winograd_enabled(self, winograd_enabled):
        self._winograd_enabled = winograd_enabled

    def disable_transpose_filters(self):
        if TransformerRule.TRANSPOSE_FILTERS in self._transformer_option:
            self._transformer_option.remove(TransformerRule.TRANSPOSE_FILTERS)

    def enable_transpose_filters(self):
        if TransformerRule.TRANSPOSE_FILTERS not in self._transformer_option:
            self._transformer_option.append(TransformerRule.TRANSPOSE_FILTERS)


class ConverterUtil(object):
    @staticmethod
    def get_arg(op, arg_name):
        for arg in op.arg:
            if arg.name == arg_name:
                return arg
        return None

    @staticmethod
    def add_data_format_arg(op, data_format):
        data_format_arg = op.arg.add()
        data_format_arg.name = MaceKeyword.mace_data_format_str
        data_format_arg.i = data_format.value

    @staticmethod
    def data_format(op):
        arg = ConverterUtil.get_arg(op, MaceKeyword.mace_data_format_str)
        if arg is None:
            return None
        elif arg.i == DataFormat.NHWC.value:
            return DataFormat.NHWC
        elif arg.i == DataFormat.NCHW.value:
            return DataFormat.NCHW
        else:
            return None

    @staticmethod
    def set_filter_format(net, filter_format):
        arg = net.arg.add()
        arg.name = MaceKeyword.mace_filter_format_str
        arg.i = filter_format.value

    @staticmethod
    def filter_format(net):
        arg = ConverterUtil.get_arg(net, MaceKeyword.mace_filter_format_str)
        if arg is None:
            return None
        elif arg.i == FilterFormat.HWIO.value:
            return FilterFormat.HWIO
        elif arg.i == FilterFormat.HWOI.value:
            return FilterFormat.HWOI
        elif arg.i == FilterFormat.OIHW.value:
            return FilterFormat.OIHW
        else:
            return None
