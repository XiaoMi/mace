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

from py_proto import mace_pb2

from utils.config_parser import DataFormat
from utils.config_parser import DeviceType
from utils.config_parser import Platform


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
    ELU = 7
    HARDSIGMOID = 8


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
    SIGN = 13
    NONE = 14
    NOT_EQUAL = 15


class ReduceType(Enum):
    MEAN = 0
    MIN = 1
    MAX = 2
    PROD = 3
    SUM = 4


class PadType(Enum):
    CONSTANT = 0
    REFLECT = 1
    SYMMETRIC = 2


class FrameworkType(Enum):
    TENSORFLOW = 0
    CAFFE = 1
    ONNX = 2
    MEGENGINE = 3
    KERAS = 4
    PYTORCH = 5


class CoordinateTransformationMode(Enum):
    NONE = 0
    HALF_PIXEL = 1
    PYTORCH_HALF_PIXEL = 2


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
    'Cumsum',
    'Deconv2D',
    'DepthToSpace',
    'DepthwiseConv2d',
    'DepthwiseDeconv2d',
    'Dequantize',
    'Eltwise',
    'ExpandDims',
    'ExtractImagePatches',
    'ExtractPooling',
    'Fill',
    'FullyConnected',
    'Gather',
    'GroupNorm',
    'Identity',
    'IfDefined',
    'InferConv2dShape',
    'InstanceNorm',
    'KaldiBatchNorm',
    'LocalResponseNorm',
    'LpNorm',
    'LSTMCell',
    'LstmNonlinear',
    'DynamicLSTM',
    'MatMul',
    'Moments',
    'MVNorm',
    'NonlocalReshape',
    'OneHot',
    'Pad',
    'PadContext',
    'PNorm',
    'Pooling',
    'PriorBox',
    'Proposal',
    'Quantize',
    'Reduce',
    'ReplaceIndex',
    'Reshape',
    'ResizeBicubic',
    'ResizeBilinear',
    'ResizeNearestNeighbor',
    'Reverse',
    'ScalarMath',
    'Select',
    'Slice',
    'Splice',
    'Split',
    'Shape',
    'Squeeze',
    'Stack',
    'Unstack',
    'Unsqueeze',
    'StridedSlice',
    'Softmax',
    'SpaceToBatchND',
    'SpaceToDepth',
    'SqrDiffMean',
    'Subsample',
    'SumGroup',
    'TargetRMSNorm',
    'Tile',
    'Transpose',
    'DetectionOutput',
    'Where',
    'LayerNorm',
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
                          MaceOp.InstanceNorm,
                          MaceOp.Pooling,
                          MaceOp.ExtractImagePatches,
                          MaceOp.ResizeBicubic,
                          MaceOp.ResizeBilinear,
                          MaceOp.ResizeNearestNeighbor,
                          MaceOp.SpaceToBatchND,
                          MaceOp.SpaceToDepth,
                          MaceOp.LpNorm,
                          MaceOp.MVNorm,
                          MaceOp.NonlocalReshape,
                          MaceOp.GroupNorm,
                          MaceOp.DetectionOutput,
                          ]

MaceTransposableDataFormatOps = [MaceOp.Activation,
                                 MaceOp.AddN,
                                 MaceOp.BiasAdd,
                                 MaceOp.ChannelShuffle,
                                 MaceOp.Concat,
                                 MaceOp.Crop,
                                 MaceOp.Eltwise,
                                 MaceOp.Pad,
                                 MaceOp.Reduce,
                                 MaceOp.Reshape,
                                 MaceOp.Softmax,
                                 MaceOp.Split,
                                 MaceOp.Squeeze,
                                 MaceOp.SqrDiffMean,
                                 MaceOp.Transpose,
                                 MaceOp.Tile]


class MaceKeyword(object):
    # node related str
    mace_input_node_name = 'mace_input_node'
    mace_output_node_name = 'mace_output_node'
    mace_buffer_type = 'buffer_type'
    # arg related str
    mace_affine_str = 'affine'
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
    mace_activation_coefficient_str = 'activation_coefficient'
    mace_hardsigmoid_alpha_str = 'hardsigmoid_alpha'
    mace_hardsigmoid_beta_str = 'hardsigmoid_beta'
    mace_resize_size_str = 'size'
    mace_batch_to_space_crops_str = 'crops'
    mace_paddings_str = 'paddings'
    mace_align_corners_str = 'align_corners'
    mace_nearest_mode_str = 'nearest_mode'
    mace_height_scale_str = 'height_scale'
    mace_width_scale_str = 'width_scale'
    mace_half_pixel_centers_str = 'half_pixel_centers'
    mace_coordinate_transformation_mode_str = 'coordinate_transformation_mode'
    mace_space_batch_block_shape_str = 'block_shape'
    mace_space_depth_block_size_str = 'block_size'
    mace_constant_value_str = 'constant_value'
    mace_dim_str = 'dim'
    mace_dims_str = 'dims'
    mace_axis_str = 'axis'
    mace_end_axis_str = 'end_axis'
    mace_num_axes_str = 'num_axes'
    mace_num_split_str = 'num_split'
    mace_size_splits_str = 'size_splits'
    mace_keepdims_str = 'keepdims'
    mace_shape_str = 'shape'
    mace_winograd_filter_transformed = 'is_filter_transformed'
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
    mace_runtime_type_str = "runtime_type"
    mace_group_str = "group"
    mace_group_num_str = "group_num"
    mace_wino_arg_str = "wino_block_size"
    mace_quantize_flag_arg_str = "quantize_flag"
    mace_epsilon_str = 'epsilon'
    mace_reduce_type_str = 'reduce_type'
    mace_argmin_str = 'argmin'
    mace_out_val_str = 'out_val'
    mace_top_k_str = 'top_k'
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
    mace_input_indexes_str = 'input_indexes'
    mace_output_indexes_str = 'output_indexes'
    mace_p_str = 'p'
    mace_nor_var_str = 'normalize_variance'
    mace_across_ch_str = 'across_channels'
    mace_apu_16bit_per_tensor = 'mace_apu_16bit_per_tensor'
    mace_apu_data_type_arg_str = 'apu_data_type'
    mace_int8 = 'int8'
    mace_num_classes = 'num_classes'
    mace_nms_threshold = 'nms_threshold'
    mace_confidence_threshold = 'confidence_threshold'
    mace_nms_top_k = 'nms_top_k'
    mace_keep_top_k = 'keep_top_k'
    mace_htp_u16a_s8w = 'mace_htp_u16a_s8w'


class QatType(Enum):
    ASYMMETRIC = 0
    SYMMETRIC = 1


class InfoKey(object):
    has_qat = 'tensors_has_qat'
    qat_type = 'tensors_qat_type_map'


class TransformerRule(Enum):
    REMOVE_USELESS_OP = 1
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
    TRANSPOSE_RESHAPE_AND_FLATTEN = 36
    FOLD_FC_RESHAPE = 37
    TRANSFORM_CHANNEL_SHUFFLE = 38
    UPDATE_DATA_FORMAT = 39
    QUANTIZE_SPECIFIC_OPS_ONLY = 40
    FP16_MATMUL_WEIGHT = 41
    FP16_GATHER_WEIGHT = 42
    QUANTIZE_LARGE_WEIGHTS = 43
    TRANSPOSE_SHAPE_TENSOR_TO_PARAM = 44
    TRANSFORM_SINGLE_BN_TO_DEPTHWISE_CONV = 45
    TRANSFORM_MUL_MAX_TO_PRELU = 46
    TRANSFORM_EXPAND_DIMS_TO_RESHAPE = 47
    QUANTIZE_FOLD_RELU = 48
    TRANSFORM_KERAS_QUANTIZE_INFO = 49
    ADD_GENERRAL_INFO = 50
    FOLD_DIV_BN = 51
    REMOVE_UNUSED_TENSOR = 52
    TRANSPOSE_CONST_OP_INPUT = 53
    FOLD_INSTANCE_NORM = 54
    FOLD_MOMENTS = 55
    UPDATE_FC_OUTPUT_SHAPE = 56
    TRANSFORM_BIASADD_TO_ADD = 57
    TRANSFORM_SLICE_TO_STRIDED_SLICE = 58
    ADD_TRANSPOSE_FOR_HTP = 59
    FOLD_LAYERNORM = 60
    FOLD_MATMUL_ADD = 61


class ConverterInterface(object):
    """Base class for converting external models to mace models."""

    def run(self):
        raise NotImplementedError('run')


class NodeInfo(object):
    """A class for describing node information"""

    def __init__(self):
        self._name = None
        self._alias = None
        self._data_type = mace_pb2.DT_FLOAT
        self._shape = []
        self._data_format = DataFormat.NHWC
        self._range = [-1.0, 1.0]

    @property
    def name(self):
        return self._name

    @property
    def alias(self):
        return self._alias

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

    @alias.setter
    def alias(self, alias):
        self._alias = alias

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
        self._name = ""
        self._order = 0
        self._input_nodes = {}
        self._output_nodes = {}
        self._check_nodes = {}
        self._data_type = mace_pb2.DT_FLOAT
        self._device = DeviceType.CPU.value
        self._winograd = 0
        self._quantize = False
        self._quantize_schema = ""
        self._quantize_large_weights = False
        self._quantize_range_file = ""
        self._change_concat_ranges = False
        self._transformer_option = None
        self._cl_mem_type = "image"
        self._quantize_stat = False
        self._platform = None
        self._enable_micro = False

    @property
    def name(self):
        return self._name

    @property
    def order(self):
        return self._order

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
    def quantize_schema(self):
        return self._quantize_schema

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

    @property
    def quantize_stat(self):
        return self._quantize_stat

    @property
    def platform(self):
        return self._platform

    @property
    def enable_micro(self):
        return self._enable_micro

    @name.setter
    def name(self, name):
        self._name = name

    @order.setter
    def order(self, order):
        self._order = order

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

    @quantize_schema.setter
    def quantize_schema(self, quantize_schema):
        self._quantize_schema = quantize_schema

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

    @quantize_stat.setter
    def quantize_stat(self, quantize_stat):
        self._quantize_stat = quantize_stat

    @platform.setter
    def platform(self, platform):
        self._platform = platform

    @enable_micro.setter
    def enable_micro(self, enable_micro):
        self._enable_micro = enable_micro

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
                TransformerRule.REMOVE_USELESS_OP,
                TransformerRule.FOLD_DIV_BN,
                TransformerRule.TRANSFORM_FAKE_QUANTIZE,
                TransformerRule.REMOVE_USELESS_OP,
                TransformerRule.TRANSFORM_GLOBAL_POOLING,
                TransformerRule.TRANSFORM_LSTMCELL_ZEROSTATE,
                TransformerRule.TRANSFORM_BASIC_LSTMCELL,
                TransformerRule.TRANSPOSE_RESHAPE_AND_FLATTEN,
                TransformerRule.TRANSPOSE_SHAPE_TENSOR_TO_PARAM,
                TransformerRule.FOLD_RESHAPE,
                TransformerRule.TRANSFORM_MATMUL_TO_FC,
                # Must be put after TRANSFORM_MATMUL_TO_FC and
                # before UPDATE_DATA_FORMAT
                TransformerRule.UPDATE_FC_OUTPUT_SHAPE,
                # After update FC output shape, there may be useless reshape
                # which does: NC11 -> Reshape(useless) -> NC11
                TransformerRule.REMOVE_USELESS_OP,
                # For StoB -> conv -> BtoS -> BN pattern
                # Insert flatten_atrous_conv before fold_xxx_and_bn
                TransformerRule.FLATTEN_ATROUS_CONV,
                TransformerRule.FOLD_BATCHNORM,
                TransformerRule.FOLD_CONV_AND_BN,
                TransformerRule.FOLD_DECONV_AND_BN,
                TransformerRule.FOLD_DEPTHWISE_CONV_AND_BN,
                TransformerRule.TRANSFORM_ADD_TO_BIASADD,
                TransformerRule.TRANSFORM_BIASADD_TO_ADD,
                TransformerRule.TRANSFORM_SLICE_TO_STRIDED_SLICE,
                TransformerRule.REARRANGE_BATCH_TO_SPACE,
                TransformerRule.FOLD_BIASADD,
                TransformerRule.FLATTEN_ATROUS_CONV,
                TransformerRule.FOLD_ACTIVATION,
                TransformerRule.FOLD_SQRDIFF_MEAN,
                # FOLD_INSTANCE_NORM depends on FOLD_SQRDIFF_MEAN
                TransformerRule.FOLD_INSTANCE_NORM,
                TransformerRule.FOLD_MOMENTS,
                TransformerRule.TRANSFORM_GLOBAL_CONV_TO_FC,
                TransformerRule.RESHAPE_FC_WEIGHT,
                TransformerRule.FOLD_FC_RESHAPE,
                TransformerRule.TRANSFORM_CHANNEL_SHUFFLE,
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
                # update the data format of ops
                TransformerRule.UPDATE_DATA_FORMAT,
                TransformerRule.TRANSPOSE_DATA_FORMAT,
                # Model data format related transformation
                TransformerRule.TRANSPOSE_FILTERS,
                TransformerRule.ADD_TRANSPOSE_FOR_HTP,
                TransformerRule.SORT_BY_EXECUTION,
                # Need to be put after SORT_BY_EXECUTION
                TransformerRule.ADD_QUANTIZE_TENSOR_RANGE,
                TransformerRule.ADD_GENERRAL_INFO,
                TransformerRule.REMOVE_UNUSED_TENSOR,
                TransformerRule.TRANSPOSE_CONST_OP_INPUT,
            ]

            if self._device == DeviceType.APU.value:
                self._transformer_option = self._transformer_option + [
                    TransformerRule.TRANSFORM_SINGLE_BN_TO_DEPTHWISE_CONV,
                    TransformerRule.TRANSFORM_MUL_MAX_TO_PRELU,
                    TransformerRule.TRANSFORM_EXPAND_DIMS_TO_RESHAPE,
                ]

            if self._device == DeviceType.HEXAGON.value:
                self._transformer_option = self._transformer_option + [
                    TransformerRule.FOLD_LAYERNORM,
                    TransformerRule.FOLD_MATMUL_ADD,
                ]
                
            if self.quantize_large_weights:
                self._transformer_option = self._transformer_option + [
                    TransformerRule.QUANTIZE_LARGE_WEIGHTS
                ]

            if self._quantize:
                self._transformer_option = self._transformer_option + [
                    # need to be put after ADD_QUANTIZE_TENSOR_RANGE
                    TransformerRule.QUANTIZE_FOLD_RELU,
                    TransformerRule.QUANTIZE_NODES,
                    TransformerRule.QUANTIZE_WEIGHTS,
                    TransformerRule.SORT_BY_EXECUTION,
                    TransformerRule.CHECK_QUANTIZE_INFO,
                ]
                if TransformerRule.TRANSFORM_GLOBAL_CONV_TO_FC in \
                        self._transformer_option:
                    self._transformer_option.remove(
                        TransformerRule.TRANSFORM_GLOBAL_CONV_TO_FC)

                if self._platform == Platform.KERAS:
                    self._transformer_option = [
                        TransformerRule.TRANSFORM_KERAS_QUANTIZE_INFO
                    ] + self._transformer_option

            if self._enable_micro:
                if TransformerRule.TRANSFORM_MATMUL_TO_FC in \
                        self._transformer_option:
                    self._transformer_option.remove(
                        TransformerRule.TRANSFORM_MATMUL_TO_FC)
                if TransformerRule.TRANSFORM_GLOBAL_CONV_TO_FC in \
                        self._transformer_option:
                    self._transformer_option.remove(
                        TransformerRule.TRANSFORM_GLOBAL_CONV_TO_FC)


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

    @staticmethod
    def set_framework_type(net, framework_type):
        framework_type_arg = net.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = framework_type

    @staticmethod
    def framework_type(net):
        framework_type_arg = ConverterUtil.get_arg(
            net, MaceKeyword.mace_framework_type_str)
        if framework_type_arg is None:
            return None
        return framework_type_arg.i
