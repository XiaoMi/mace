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


from enum import Enum

from mace.proto import mace_pb2


class DeviceType(Enum):
    CPU = 0
    GPU = 2
    HEXAGON = 3
    HTA = 4
    APU = 5


class DataFormat(Enum):
    NONE = 0
    NHWC = 1
    NCHW = 2
    HWIO = 100
    OIHW = 101
    HWOI = 102
    OHWI = 103
    AUTO = 1000


# SAME_LOWER: if the amount of paddings to be added is odd,
# it will add the extra data to the right or bottom
class PaddingMode(Enum):
    VALID = 0
    SAME = 1
    FULL = 2
    SAME_LOWER = 3
    NA = 4


class PoolingType(Enum):
    AVG = 1
    MAX = 2


class RoundMode(Enum):
    FLOOR = 0
    CEIL = 1


class ActivationType(Enum):
    NOOP = 0
    RELU = 1
    RELUX = 2
    PRELU = 3
    TANH = 4
    SIGMOID = 5
    LEAKYRELU = 6


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
    EQUAL = 10
    FLOOR_DIV = 11
    CLIP = 12


class ReduceType(Enum):
    MEAN = 0
    MIN = 1
    MAX = 2
    PROD = 3


class PadType(Enum):
    CONSTANT = 0
    REFLECT = 1
    SYMMETRIC = 2


class FrameworkType(Enum):
    TENSORFLOW = 0
    CAFFE = 1
    ONNX = 2


MaceSupportedOps = [
    'Activation',
    'AddN',
    'ArgMax',
    'BatchNorm',
    'BatchToSpaceND',
    'BiasAdd',
    'Cast',
    'ChannelShuffle',
    'Concat',
    'Conv2D',
    'Crop',
    'Deconv2D',
    'Delay',
    'DepthToSpace',
    'DepthwiseConv2d',
    'DepthwiseDeconv2d',
    'Dequantize',
    'Eltwise',
    'ExpandDims',
    'ExtractPooling',
    'Fill',
    'FullyConnected',
    'Gather',
    'Identity',
    'InferConv2dShape',
    'KaldiBatchNorm',
    'LocalResponseNorm',
    'LSTMCell',
    'LstmNonlinear',
    'DynamicLSTM',
    'MatMul',
    'OneHot',
    'Pad',
    'PadContext',
    'PNorm',
    'Pooling',
    'PriorBox',
    'Proposal',
    'Quantize',
    'Reduce',
    'Reshape',
    'ResizeBicubic',
    'ResizeBilinear',
    'ResizeNearestNeighbor',
    'Reverse',
    'ScalarMath',
    'Slice',
    'Splice',
    'Split',
    'Shape',
    'Squeeze',
    'Stack',
    'Unstack',
    'StridedSlice',
    'Softmax',
    'SpaceToBatchND',
    'SpaceToDepth',
    'SqrDiffMean',
    'SumGroup',
    'TargetRMSNorm',
    'Transpose',
    'Cumsum',
]

MaceOp = Enum('MaceOp', [(op, op) for op in MaceSupportedOps], type=str)

MaceFixedDataFormatOps = [MaceOp.BatchNorm,
                          MaceOp.BatchToSpaceND,
                          MaceOp.Conv2D,
                          MaceOp.Deconv2D,
                          MaceOp.DepthToSpace,
                          MaceOp.DepthwiseConv2d,
                          MaceOp.DepthwiseDeconv2d,
                          MaceOp.FullyConnected,
                          MaceOp.Pooling,
                          MaceOp.ResizeBicubic,
                          MaceOp.ResizeBilinear,
                          MaceOp.ResizeNearestNeighbor,
                          MaceOp.SpaceToBatchND,
                          MaceOp.SpaceToDepth]

MaceTransposableDataFormatOps = [MaceOp.Activation,
                                 MaceOp.AddN,
                                 MaceOp.BiasAdd,
                                 MaceOp.ChannelShuffle,
                                 MaceOp.Concat,
                                 MaceOp.Crop,
                                 MaceOp.Eltwise,
                                 MaceOp.Pad,
                                 MaceOp.Reduce,
                                 MaceOp.Softmax,
                                 MaceOp.Split,
                                 MaceOp.SqrDiffMean]


class MaceKeyword(object):
    # node related str
    mace_input_node_name = 'mace_input_node'
    mace_output_node_name = 'mace_output_node'
    mace_buffer_type = 'buffer_type'
    # arg related str
    mace_padding_str = 'padding'
    mace_padding_type_str = 'padding'
    mace_padding_values_str = 'padding_values'
    mace_strides_str = 'strides'
    mace_dilations_str = 'dilations'
    mace_pooling_type_str = 'pooling_type'
    mace_global_pooling_str = 'global_pooling'
    mace_kernel_str = 'kernels'
    mace_data_format_str = 'data_format'
    mace_has_data_format_str = 'has_data_format'
    mace_filter_format_str = 'filter_format'
    mace_element_type_str = 'type'
    mace_activation_type_str = 'activation'
    mace_activation_max_limit_str = 'max_limit'
    mace_activation_leakyrelu_coefficient_str = 'leakyrelu_coefficient'
    mace_resize_size_str = 'size'
    mace_batch_to_space_crops_str = 'crops'
    mace_paddings_str = 'paddings'
    mace_align_corners_str = 'align_corners'
    mace_space_batch_block_shape_str = 'block_shape'
    mace_space_depth_block_size_str = 'block_size'
    mace_constant_value_str = 'constant_value'
    mace_dim_str = 'dim'
    mace_dims_str = 'dims'
    mace_axis_str = 'axis'
    mace_end_axis_str = 'end_axis'
    mace_num_axes_str = 'num_axes'
    mace_num_split_str = 'num_split'
    mace_keepdims_str = 'keepdims'
    mace_shape_str = 'shape'
    mace_winograd_filter_transformed = 'is_filter_transformed'
    mace_device = 'device'
    mace_scalar_input_str = 'scalar_input'
    mace_wino_block_size = 'wino_block_size'
    mace_output_shape_str = 'output_shape'
    mace_begin_mask_str = 'begin_mask'
    mace_end_mask_str = 'end_mask'
    mace_ellipsis_mask_str = 'ellipsis_mask'
    mace_new_axis_mask_str = 'new_axis_mask'
    mace_shrink_axis_mask_str = 'shrink_axis_mask'
    mace_transpose_a_str = 'transpose_a'
    mace_transpose_b_str = 'transpose_b'
    mace_op_data_type_str = 'T'
    mace_offset_str = 'offset'
    mace_opencl_max_image_size = "opencl_max_image_size"
    mace_seperate_buffer_str = 'seperate_buffer'
    mace_scalar_input_index_str = 'scalar_input_index'
    mace_opencl_mem_type = "opencl_mem_type"
    mace_framework_type_str = "framework_type"
    mace_group_str = "group"
    mace_wino_arg_str = "wino_block_size"
    mace_quantize_flag_arg_str = "quantize_flag"
    mace_epsilon_str = 'epsilon'
    mace_reduce_type_str = 'reduce_type'
    mace_argmin_str = 'argmin'
    mace_round_mode_str = 'round_mode'
    mace_min_size_str = 'min_size'
    mace_max_size_str = 'max_size'
    mace_aspect_ratio_str = 'aspect_ratio'
    mace_flip_str = 'flip'
    mace_clip_str = 'clip'
    mace_variance_str = 'variance'
    mace_step_h_str = 'step_h'
    mace_step_w_str = 'step_w'
    mace_find_range_every_time = 'find_range_every_time'
    mace_non_zero = 'non_zero'
    mace_pad_type_str = 'pad_type'
    mace_exclusive_str = 'exclusive'
    mace_reverse_str = 'reverse'
    mace_const_data_num_arg_str = 'const_data_num'
    mace_coeff_str = 'coeff'


class TransformerRule(Enum):
    REMOVE_IDENTITY_OP = 1
    TRANSFORM_GLOBAL_POOLING = 2
    FOLD_RESHAPE = 3
    TRANSFORM_MATMUL_TO_FC = 4
    FOLD_BATCHNORM = 5
    FOLD_CONV_AND_BN = 6
    FOLD_DEPTHWISE_CONV_AND_BN = 7
    ADD_WINOGRAD_ARG = 8
    TRANSFORM_ADD_TO_BIASADD = 9
    FOLD_BIASADD = 10
    FLATTEN_ATROUS_CONV = 11
    FOLD_ACTIVATION = 12
    TRANSPOSE_FILTERS = 13
    RESHAPE_FC_WEIGHT = 14
    TRANSPOSE_DATA_FORMAT = 15
    TRANSFORM_GLOBAL_CONV_TO_FC = 16
    ADD_BUFFER_TRANSFORM = 17
    ADD_DEVICE = 18
    SORT_BY_EXECUTION = 19
    ADD_IN_OUT_TENSOR_INFO = 20
    ADD_MACE_INPUT_AND_OUTPUT_NODES = 21
    UPDATE_FLOAT_OP_DATA_TYPE = 22
    QUANTIZE_NODES = 23
    ADD_QUANTIZE_TENSOR_RANGE = 24
    QUANTIZE_WEIGHTS = 25
    TRANSFORM_LSTMCELL_ZEROSTATE = 26
    TRANSFORM_BASIC_LSTMCELL = 27
    TRANSFORM_FAKE_QUANTIZE = 28
    CHECK_QUANTIZE_INFO = 29
    REARRANGE_BATCH_TO_SPACE = 30
    ADD_OPENCL_INFORMATIONS = 31
    FOLD_DECONV_AND_BN = 32
    FOLD_SQRDIFF_MEAN = 33
    TRANSPOSE_MATMUL_WEIGHT = 34
    FOLD_EMBEDDING_LOOKUP = 35
    TRANSPOSE_CAFFE_RESHAPE_AND_FLATTEN = 36
    FOLD_FC_RESHAPE = 37
    TRANSFORM_CHANNEL_SHUFFLE = 38
    UPDATE_DATA_FORMAT = 39
    QUANTIZE_SPECIFIC_OPS_ONLY = 40
    FP16_MATMUL_WEIGHT = 41
    FP16_GATHER_WEIGHT = 42
    QUANTIZE_LARGE_WEIGHTS = 43


class ConverterInterface(object):
    """Base class for converting external models to mace models."""

    def run(self):
        raise NotImplementedError('run')


class NodeInfo(object):
    """A class for describing node information"""

    def __init__(self):
        self._name = None
        self._data_type = mace_pb2.DT_FLOAT
        self._shape = []
        self._data_format = DataFormat.NHWC
        self._range = [-1.0, 1.0]

    @property
    def name(self):
        return self._name

    @property
    def data_type(self):
        return self._data_type

    @property
    def shape(self):
        return self._shape

    @property
    def data_format(self):
        return self._data_format

    @property
    def range(self):
        return self._range

    @name.setter
    def name(self, name):
        self._name = name

    @data_type.setter
    def data_type(self, data_type):
        self._data_type = data_type

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @data_format.setter
    def data_format(self, data_format):
        self._data_format = data_format

    @range.setter
    def range(self, range):
        self._range = range

    def __str__(self):
        return '%s %s' % (self._name, str(self._shape))


class ConverterOption(object):
    """A class for specifying options passed to converter tool"""

    def __init__(self):
        self._input_nodes = {}
        self._output_nodes = {}
        self._check_nodes = {}
        self._data_type = mace_pb2.DT_FLOAT
        self._device = DeviceType.CPU.value
        self._winograd = 0
        self._quantize = False
        self._quantize_large_weights = False
        self._quantize_range_file = ""
        self._change_concat_ranges = False
        self._transformer_option = None
        self._cl_mem_type = ""

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def output_nodes(self):
        return self._output_nodes

    @property
    def check_nodes(self):
        return self._check_nodes

    @property
    def data_type(self):
        return self._data_type

    @property
    def device(self):
        return self._device

    @property
    def winograd(self):
        return self._winograd

    @property
    def quantize(self):
        return self._quantize

    @property
    def quantize_large_weights(self):
        return self._quantize_large_weights

    @property
    def change_concat_ranges(self):
        return self._change_concat_ranges

    @property
    def quantize_range_file(self):
        return self._quantize_range_file

    @property
    def transformer_option(self):
        return self._transformer_option

    @property
    def cl_mem_type(self):
        return self._cl_mem_type

    @input_nodes.setter
    def input_nodes(self, input_nodes):
        for node in input_nodes.values():
            self._input_nodes[node.name] = node

    def add_input_node(self, input_node):
        self._input_nodes[input_node.name] = input_node

    @output_nodes.setter
    def output_nodes(self, output_nodes):
        for node in output_nodes.values():
            self.output_nodes[node.name] = node

    def add_output_node(self, output_node):
        self._output_nodes[output_node.name] = output_node

    @check_nodes.setter
    def check_nodes(self, check_nodes):
        for node in check_nodes.values():
            self.check_nodes[node.name] = node

    def add_check_node(self, check_node):
        self._check_nodes[check_node.name] = check_node

    @data_type.setter
    def data_type(self, data_type):
        self._data_type = data_type

    @device.setter
    def device(self, device):
        self._device = device

    @winograd.setter
    def winograd(self, winograd):
        self._winograd = winograd

    @quantize.setter
    def quantize(self, quantize):
        self._quantize = quantize

    @quantize_large_weights.setter
    def quantize_large_weights(self, quantize_large_weights):
        self._quantize_large_weights = quantize_large_weights

    @quantize_range_file.setter
    def quantize_range_file(self, quantize_range_file):
        self._quantize_range_file = quantize_range_file

    @change_concat_ranges.setter
    def change_concat_ranges(self, change_concat_ranges):
        self._change_concat_ranges = change_concat_ranges

    @transformer_option.setter
    def transformer_option(self, transformer_option):
        self._transformer_option = transformer_option

    @cl_mem_type.setter
    def cl_mem_type(self, cl_mem_type):
        self._cl_mem_type = cl_mem_type

    def disable_transpose_filters(self):
        if TransformerRule.TRANSPOSE_FILTERS in self._transformer_option:
            self._transformer_option.remove(TransformerRule.TRANSPOSE_FILTERS)

    def enable_transpose_filters(self):
        if TransformerRule.TRANSPOSE_FILTERS not in self._transformer_option:
            self._transformer_option.append(TransformerRule.TRANSPOSE_FILTERS)

    def build(self):
        if self._transformer_option:
            self._transformer_option = [TransformerRule[transformer]
                                        for transformer in self._transformer_option]  # noqa
        else:
            self._transformer_option = [
                # Model structure related transformation
                TransformerRule.TRANSFORM_FAKE_QUANTIZE,
                TransformerRule.REMOVE_IDENTITY_OP,
                TransformerRule.TRANSFORM_GLOBAL_POOLING,
                TransformerRule.TRANSFORM_LSTMCELL_ZEROSTATE,
                TransformerRule.TRANSFORM_BASIC_LSTMCELL,
                TransformerRule.TRANSPOSE_CAFFE_RESHAPE_AND_FLATTEN,
                TransformerRule.FOLD_RESHAPE,
                TransformerRule.TRANSFORM_MATMUL_TO_FC,
                # For StoB -> conv -> BtoS -> BN pattern
                # Insert flatten_atrous_conv before fold_xxx_and_bn
                TransformerRule.FLATTEN_ATROUS_CONV,
                TransformerRule.FOLD_BATCHNORM,
                TransformerRule.FOLD_CONV_AND_BN,
                TransformerRule.FOLD_DECONV_AND_BN,
                TransformerRule.FOLD_DEPTHWISE_CONV_AND_BN,
                TransformerRule.TRANSFORM_ADD_TO_BIASADD,
                TransformerRule.REARRANGE_BATCH_TO_SPACE,
                TransformerRule.FOLD_BIASADD,
                TransformerRule.FLATTEN_ATROUS_CONV,
                TransformerRule.FOLD_ACTIVATION,
                TransformerRule.FOLD_SQRDIFF_MEAN,
                TransformerRule.TRANSFORM_GLOBAL_CONV_TO_FC,
                TransformerRule.RESHAPE_FC_WEIGHT,
                TransformerRule.FOLD_FC_RESHAPE,
                TransformerRule.TRANSFORM_CHANNEL_SHUFFLE,
                # Model data format related transformation
                TransformerRule.TRANSPOSE_FILTERS,
                # Mace model structure related transformation
                TransformerRule.ADD_IN_OUT_TENSOR_INFO,
                TransformerRule.TRANSPOSE_MATMUL_WEIGHT,
                # Add winograd argument
                TransformerRule.ADD_WINOGRAD_ARG,
                # Data type related transformation
                TransformerRule.UPDATE_FLOAT_OP_DATA_TYPE,
                # Transform finalization
                TransformerRule.ADD_OPENCL_INFORMATIONS,
                # for quantization entropy calibration use
                TransformerRule.SORT_BY_EXECUTION,
                # update the data format of ops
                TransformerRule.UPDATE_DATA_FORMAT,
                TransformerRule.TRANSPOSE_DATA_FORMAT,
                # Need to be put after SORT_BY_EXECUTION
                TransformerRule.ADD_QUANTIZE_TENSOR_RANGE,
            ]
            if self.quantize_large_weights:
                self._transformer_option = self._transformer_option + [
                    TransformerRule.QUANTIZE_LARGE_WEIGHTS
                ]
            if self._quantize:
                self._transformer_option = self._transformer_option + [
                    # need to be put after ADD_QUANTIZE_TENSOR_RANGE
                    TransformerRule.QUANTIZE_NODES,
                    TransformerRule.QUANTIZE_WEIGHTS,
                    TransformerRule.SORT_BY_EXECUTION,
                    TransformerRule.CHECK_QUANTIZE_INFO,
                ]


class ConverterUtil(object):
    @staticmethod
    def get_arg(op, arg_name):
        for arg in op.arg:
            if arg.name == arg_name:
                return arg
        return None

    @staticmethod
    def del_arg(op, arg_name):
        found_idx = -1
        for idx in range(len(op.arg)):
            if op.arg[idx].name == arg_name:
                found_idx = idx
                break
        if found_idx != -1:
            del op.arg[found_idx]

    @staticmethod
    def add_data_format_arg(op, data_format):
        data_format_arg = op.arg.add()
        data_format_arg.name = MaceKeyword.mace_data_format_str
        data_format_arg.i = data_format.value

    @staticmethod
    def add_data_type_arg(op, data_type):
        data_type_arg = op.arg.add()
        data_type_arg.name = MaceKeyword.mace_op_data_type_str
        data_type_arg.i = data_type

    @staticmethod
    def data_format(op):
        arg = ConverterUtil.get_arg(op, MaceKeyword.mace_data_format_str)
        if arg is None:
            return None
        elif arg.i == DataFormat.NHWC.value:
            return DataFormat.NHWC
        elif arg.i == DataFormat.NCHW.value:
            return DataFormat.NCHW
        elif arg.i == DataFormat.AUTO.value:
            return DataFormat.AUTO
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
        elif arg.i == DataFormat.HWIO.value:
            return DataFormat.HWIO
        elif arg.i == DataFormat.HWOI.value:
            return DataFormat.HWOI
        elif arg.i == DataFormat.OIHW.value:
            return DataFormat.OIHW
        else:
            return None
