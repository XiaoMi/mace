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


import re

import numpy as np
import six

from py_proto import mace_pb2
from transform import base_converter
from transform.base_converter import ActivationType
from transform.base_converter import ConverterUtil
from transform.base_converter import DataFormat
from transform.base_converter import DeviceType
from transform.base_converter import EltwiseType
from transform.base_converter import FrameworkType
from transform.base_converter import InfoKey
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp
from transform.base_converter import MaceFixedDataFormatOps
from transform.base_converter import MaceTransposableDataFormatOps
from transform.base_converter import PaddingMode
from transform.base_converter import QatType
from transform.base_converter import ReduceType
from transform.base_converter import TransformerRule
from utils.config_parser import MemoryType
from utils.config_parser import Platform
from quantize import quantize_util
from utils.util import mace_check
from validate import calculate_similarity


class Transformer(base_converter.ConverterInterface):
    """A class for transform naive mace model to optimized model.
    This Transformer should be platform irrelevant. So, do not assume
    tensor name has suffix like ':0".
    """

    def __init__(self, option, model, converter_info):
        # Dependencies
        # (TRANSFORM_MATMUL_TO_FC, TRANSFORM_GLOBAL_CONV_TO_FC) -> RESHAPE_FC_WEIGHT  # noqa
        self._registered_transformers = {
            TransformerRule.TRANSFORM_FAKE_QUANTIZE:
                self.transform_fake_quantize,
            TransformerRule.REMOVE_USELESS_OP: self.remove_useless_op,
            TransformerRule.FOLD_DIV_BN: self.fold_div_bn,
            TransformerRule.TRANSPOSE_CONST_OP_INPUT:
                self.transpose_const_op_input,
            TransformerRule.TRANSFORM_GLOBAL_POOLING:
                self.transform_global_pooling,
            TransformerRule.TRANSFORM_LSTMCELL_ZEROSTATE:
                self.transform_lstmcell_zerostate,
            TransformerRule.TRANSFORM_BASIC_LSTMCELL:
                self.transform_basic_lstmcell,
            TransformerRule.FOLD_RESHAPE: self.fold_reshape,
            TransformerRule.TRANSFORM_MATMUL_TO_FC:
                self.transform_matmul_to_fc,
            TransformerRule.UPDATE_FC_OUTPUT_SHAPE:
                self.update_fc_output_shape,
            TransformerRule.FOLD_BATCHNORM: self.fold_batchnorm,
            TransformerRule.FOLD_BIASADD: self.fold_biasadd,
            TransformerRule.FOLD_CONV_AND_BN:
                self.fold_conv_and_bn,  # data_format related
            TransformerRule.FOLD_DECONV_AND_BN:
                self.fold_deconv_and_bn,  # data_format related
            TransformerRule.FOLD_DEPTHWISE_CONV_AND_BN:
                self.fold_depthwise_conv_and_bn,  # data_format related
            TransformerRule.TRANSFORM_ADD_TO_BIASADD:
                self.transform_add_to_biasadd,
            TransformerRule.TRANSFORM_BIASADD_TO_ADD:
                self.transform_biasadd_to_add,
            TransformerRule.REARRANGE_BATCH_TO_SPACE:
                self.rearrange_batch_to_space,
            TransformerRule.FLATTEN_ATROUS_CONV: self.flatten_atrous_conv,
            TransformerRule.FOLD_ACTIVATION: self.fold_activation,
            TransformerRule.FOLD_SQRDIFF_MEAN: self.fold_squared_diff_mean,
            # fold_instance_norm depends on fold_squared_diff_mean
            TransformerRule.FOLD_INSTANCE_NORM: self.fold_instance_norm,
            TransformerRule.FOLD_MOMENTS: self.fold_moments,
            TransformerRule.FOLD_EMBEDDING_LOOKUP: self.fold_embedding_lookup,
            TransformerRule.TRANSPOSE_FILTERS: self.transpose_filters,
            TransformerRule.TRANSPOSE_MATMUL_WEIGHT:
                self.transpose_matmul_weight,
            TransformerRule.FOLD_FC_RESHAPE:
                self.fold_fc_reshape,
            TransformerRule.ADD_IN_OUT_TENSOR_INFO:
                self.add_in_out_tensor_info,
            TransformerRule.ADD_WINOGRAD_ARG: self.add_winograd_arg,
            TransformerRule.TRANSFORM_GLOBAL_CONV_TO_FC:
                self.transform_global_conv_to_fc,
            TransformerRule.RESHAPE_FC_WEIGHT: self.reshape_fc_weight,
            TransformerRule.QUANTIZE_NODES:
                self.quantize_nodes,
            TransformerRule.ADD_QUANTIZE_TENSOR_RANGE:
                self.add_quantize_tensor_range,
            TransformerRule.QUANTIZE_WEIGHTS:
                self.quantize_weights,
            TransformerRule.UPDATE_FLOAT_OP_DATA_TYPE:
                self.update_float_op_data_type,
            TransformerRule.ADD_OPENCL_INFORMATIONS:
                self.add_opencl_informations,
            TransformerRule.SORT_BY_EXECUTION: self.sort_by_execution,
            TransformerRule.UPDATE_DATA_FORMAT: self.update_data_format,
            TransformerRule.TRANSPOSE_RESHAPE_AND_FLATTEN:
                self.transform_reshape_and_flatten,
            TransformerRule.TRANSPOSE_SHAPE_TENSOR_TO_PARAM:
                self.transform_shape_tensor_to_param,
            TransformerRule.TRANSPOSE_DATA_FORMAT: self.transpose_data_format,
            TransformerRule.CHECK_QUANTIZE_INFO:
                self.check_quantize_info,
            TransformerRule.TRANSFORM_CHANNEL_SHUFFLE:
                self.transform_channel_shuffle,
            TransformerRule.QUANTIZE_SPECIFIC_OPS_ONLY:
                self.quantize_specific_ops_only,
            TransformerRule.FP16_MATMUL_WEIGHT:
                self.fp16_matmul_weight,
            TransformerRule.FP16_GATHER_WEIGHT:
                self.fp16_gather_weight,
            TransformerRule.QUANTIZE_LARGE_WEIGHTS:
                self.quantize_large_weights,
            TransformerRule.TRANSFORM_SINGLE_BN_TO_DEPTHWISE_CONV:
                self.transform_single_bn_to_depthwise_conv,
            TransformerRule.TRANSFORM_MUL_MAX_TO_PRELU:
                self.transform_mul_max_to_prelu,
            TransformerRule.TRANSFORM_EXPAND_DIMS_TO_RESHAPE:
                self.transform_expand_dims_to_reshape,
            TransformerRule.QUANTIZE_FOLD_RELU:
                self.quantize_fold_relu,
            TransformerRule.TRANSFORM_KERAS_QUANTIZE_INFO:
                self.transform_keras_quantize_info,
            TransformerRule.ADD_GENERRAL_INFO:
                self.add_general_info,
            TransformerRule.REMOVE_UNUSED_TENSOR:
                self.remove_unused_tensor,
            TransformerRule.TRANSFORM_SLICE_TO_STRIDED_SLICE:
                self.transform_slice_to_strided_slice,
            TransformerRule.ADD_TRANSPOSE_FOR_HTP:
                self.add_transpose_for_htp
        }

        self._option = option
        self._model = model
        self._wino_arg = self._option.winograd

        self._ops = {}
        self._consts = {}
        self._consumers = {}
        self._producer = {}
        self._quantize_activation_info = {}
        self._quantized_tensor = set()

        self.input_name_map = {}
        self.output_name_map = {}
        self._has_none_df = False
        self.initialize_name_map()
        self._converter_info = converter_info

    def run(self):
        for key in self._option.transformer_option:
            transformer = self._registered_transformers[key]
            while True:
                self.construct_ops_and_consumers(key)
                changed = transformer()
                if not changed:
                    break
        return self._model, self._quantize_activation_info

    def initialize_name_map(self):
        for input_node in self._option.input_nodes.values():
            # When tf.Keras > 2.2 version, input_node, it is possible
            # that input_node.name and model input tensor name are different.
            if self._option.platform == Platform.KERAS:
                input_name_parts = input_node.name.split(":")
                if len(input_name_parts) == 2:
                    input_name_without_postfix = input_name_parts[0]
                    for op in self._model.op:
                        for i, name in enumerate(op.input):
                            if name == input_name_without_postfix:
                                op.input[i] = input_node.name
            new_input_name = (MaceKeyword.mace_input_node_name
                              + '_' + input_node.name)
            self.input_name_map[input_node.name] = new_input_name
            if input_node.data_type == mace_pb2.DT_INT32:
                self.input_name_map[input_node.name] = input_node.name
            if input_node.data_format == DataFormat.NONE:
                self._has_none_df = True

        output_nodes = self._option.check_nodes.values()
        for output_node in output_nodes:
            new_output_name = MaceKeyword.mace_output_node_name \
                              + '_' + output_node.name
            self.output_name_map[output_node.name] = new_output_name

    def filter_format(self):
        filter_format_value = ConverterUtil.get_arg(self._model,
                                                    MaceKeyword.mace_filter_format_str).i  # noqa
        filter_format = None
        if filter_format_value == DataFormat.HWIO.value:
            filter_format = DataFormat.HWIO
        elif filter_format_value == DataFormat.OIHW.value:
            filter_format = DataFormat.OIHW
        elif filter_format_value == DataFormat.HWOI.value:
            filter_format = DataFormat.HWOI
        else:
            mace_check(False, "filter format %d not supported" %
                       filter_format_value)

        return filter_format

    def set_filter_format(self, filter_format):
        arg = ConverterUtil.get_arg(self._model,
                                    MaceKeyword.mace_filter_format_str)
        arg.i = filter_format.value

    def construct_ops_and_consumers(self, key):
        self._ops.clear()
        self._consumers.clear()
        self._producer.clear()
        for op in self._model.op:
            self._ops[op.name] = op
        for tensor in self._model.tensors:
            self._consts[tensor.name] = tensor
        for op in self._ops.values():
            for input_tensor in op.input:
                if input_tensor not in self._consumers:
                    self._consumers[input_tensor] = []
                self._consumers[input_tensor].append(op)

            for output_tensor in op.output:
                self._producer[output_tensor] = op
        if key != TransformerRule.SORT_BY_EXECUTION:
            for input_node in self._option.input_nodes.values():
                input_node_existed = False
                for op in self._model.op:
                    if input_node.name in op.output:
                        input_node_existed = True
                        break
                if not input_node_existed:
                    op = mace_pb2.OperatorDef()
                    op.name = self.normalize_op_name(input_node.name)
                    op.type = "Input"
                    data_type_arg = op.arg.add()
                    data_type_arg.name = MaceKeyword.mace_op_data_type_str
                    data_type_arg.i = input_node.data_type
                    op.output.extend([input_node.name])
                    output_shape = op.output_shape.add()
                    output_shape.dims.extend(input_node.shape)
                    if input_node.data_format != DataFormat.NONE:
                        if input_node.data_format == DataFormat.NCHW:
                            self.transpose_shape(output_shape.dims,
                                                 [0, 3, 1, 2])
                        ConverterUtil.add_data_format_arg(op,
                                                          DataFormat.AUTO)
                    else:
                        ConverterUtil.add_data_format_arg(op,
                                                          DataFormat.NONE)
                    self._producer[op.output[0]] = op

    @staticmethod
    def replace(obj_list, source, target):
        for i in six.moves.range(len(obj_list)):
            if obj_list[i] == source:
                obj_list[i] = target

    @staticmethod
    def transpose_shape(shape, order):
        transposed_shape = []
        for i in six.moves.range(len(order)):
            transposed_shape.append(shape[order[i]])
        shape[:] = transposed_shape[:]

    @staticmethod
    def normalize_op_name(name):
        return name.replace(':', '_')

    def get_tensor_shape(self, tensor):
        if tensor in self._consts:
            return list(self._consts[tensor].dims)
        elif tensor in self._producer:
            producer = self._producer[tensor]
            for i in six.moves.range(len(producer.output)):
                if producer.output[i] == tensor:
                    return list(producer.output_shape[i].dims)
        else:
            return None

    def get_tensor_data_type(self, tensor):
        if tensor in self._consts:
            return self._consts[tensor].data_type
        elif tensor in self._producer:
            producer = self._producer[tensor]
            for i in six.moves.range(len(producer.output)):
                if producer.output[i] == tensor:
                    if i < len(producer.output_type):
                        return producer.output_type[i]
                    elif ConverterUtil.get_arg(producer, "T") is not None:
                        return ConverterUtil.get_arg(producer, "T").i
                    else:
                        print("No data type filled: ", producer)
                        return None
        else:
            return None

    def get_tensor_data_format(self, tensor):
        if tensor in self._producer:
            producer = self._producer[tensor]
            return ConverterUtil.data_format(producer)
        else:
            return DataFormat.NONE

    def consumer_count(self, tensor_name):
        return len(self._consumers.get(tensor_name, []))

    def is_op_output_node(self, op):
        output_node_tensor_names = [out for out in
                                    self._option.output_nodes]
        for output in op.output:
            if output in output_node_tensor_names:
                return True

        return False

    def safe_remove_node(self, op, replace_op, remove_input_tensor=False):
        """remove op.
        1. change the inputs of its consumers to the outputs of replace_op
        2. if the op is output node, change output node to replace op"""

        if replace_op is None:
            # When no replace op specified, we change the inputs of
            # its consumers to the input of the op. This handles the case
            # that the op is identity op and its input is a tensor.
            reshape_const_dim = op.type == MaceOp.Reshape.name and \
                (len(op.input) == 1 or op.input[1] in self._consts)

            mace_check(len(op.output) == 1 and len(op.input) == 1 or
                       reshape_const_dim,
                       "cannot remove op that w/o replace op specified"
                       " and input/output length > 1\n" + str(op))

            for consumer_op in self._consumers.get(op.output[0], []):
                self.replace(consumer_op.input, op.output[0], op.input[0])

            mace_check(op.output[0] not in self._option.output_nodes,
                       "cannot remove op that is output node")
        else:
            mace_check(len(op.output) == len(replace_op.output),
                       "cannot remove op since len(op.output) "
                       "!= len(replace_op.output)")

            for i in six.moves.range(len(op.output)):
                # if the op is output node, change replace_op output name
                # to the op output name
                if op.output[i] in self._option.output_nodes:
                    for consumer in self._consumers.get(
                            replace_op.output[i], []):
                        self.replace(consumer.input,
                                     replace_op.output[i],
                                     op.output[i])
                    replace_op.output[i] = op.output[i]
                else:
                    for consumer_op in self._consumers.get(op.output[i], []):
                        self.replace(consumer_op.input,
                                     op.output[i],
                                     replace_op.output[i])

        if remove_input_tensor:
            for input_name in op.input:
                if input_name in self._consts:
                    const_tensor = self._consts[input_name]
                    self._model.tensors.remove(const_tensor)

        self._model.op.remove(op)

    def add_in_out_tensor_info(self):
        net = self._model
        for input_node in self._option.input_nodes.values():
            input_info = net.input_info.add()
            input_info.name = input_node.name
            if input_node.alias is not None:
                input_info.alias = input_node.alias
            else:
                input_info.alias = input_node.name
            input_info.data_format = input_node.data_format.value
            input_info.dims.extend(input_node.shape)
            input_info.data_type = input_node.data_type

        # tools/python/convert.py sets option.check_nodes
        output_nodes = self._option.check_nodes.values()
        for output_node in output_nodes:
            output_info = net.output_info.add()
            output_info.name = output_node.name
            if output_node.alias is not None:
                output_info.alias = output_node.alias
            else:
                output_info.alias = output_node.name
            output_info.data_format = output_node.data_format.value
            output_info.dims.extend(output_node.shape)
            output_info.data_type = output_node.data_type

        return False

    def remove_useless_op(self):
        net = self._model
        for op in net.op:
            if self.is_op_output_node(op) and \
                    self._option.device == DeviceType.CPU.value:
                continue
            if op.type == 'Identity':
                print("Remove useless op: %s(%s)" % (op.name, op.type))
                self.safe_remove_node(op,
                                      self._producer.get(op.input[0], None))
                return True
            elif op.type == 'Reshape' and len(op.output_shape) == 1 and \
                    op.output_shape[0].dims == \
                    self.get_tensor_shape(op.input[0]):
                mace_check(len(op.output_shape[0].dims) != 0,
                           "Output shape is null in op: %s(%s)"
                           % (op.name, op.type))
                print("Remove useless reshape: %s(%s)" % (op.name, op.type))
                self.safe_remove_node(op,
                                      self._producer.get(op.input[0], None))
                return True
            elif op.type == 'Eltwise' and \
                    ConverterUtil.get_arg(
                        op, MaceKeyword.mace_element_type_str).i == \
                    EltwiseType.PROD.value:
                scala = ConverterUtil.get_arg(
                                    op, MaceKeyword.mace_scalar_input_str)
                if scala is not None and scala.f == 1.0:
                    print("Remove useless eltwise mul: %s(%s)" % (op.name,
                                                                  op.type))
                    self.safe_remove_node(op,
                                          self._producer.get(op.input[0],
                                                             None))
                    return True
            elif op.type == 'Reshape' and len(op.output_shape) == 1 and \
                    self._producer.get(op.input[0], None) is not None and \
                    self._producer.get(op.input[0], None).type == 'Reshape':
                print("Remove useless Reshape: %s(%s)" % (op.name, op.type))
                producer_op = self._producer.get(op.input[0], None)
                if (len(producer_op.input) == 1 or producer_op.input[1] in self._consts) and \
                        self._consumers.get(producer_op.output[0], None) is not None and \
                        len(self._consumers.get(producer_op.output[0], None)) == 1:
                    self.safe_remove_node(producer_op, None,
                                          remove_input_tensor=True)
                    return True
        return False

    def transform_global_pooling(self):
        net = self._model
        for op in net.op:
            if op.type == MaceOp.Pooling.name and \
                            ConverterUtil.get_arg(op,
                                                  MaceKeyword.mace_global_pooling_str) is not None:  # noqa
                print("Transform global pooling: %s(%s)" % (op.name, op.type))
                input_shape = self._producer[op.input[0]].output_shape[0].dims
                if ConverterUtil.data_format(op) == DataFormat.NHWC:
                    kernel_shape = input_shape[1:3]
                else:
                    kernel_shape = input_shape[2:4]
                ConverterUtil.get_arg(op,
                                      MaceKeyword.mace_kernel_str).ints[:] \
                    = kernel_shape[:]

        return False

    def fold_batchnorm(self):
        net = self._model
        for op in net.op:
            if not self._has_none_df and \
                    (op.type == MaceOp.Eltwise.name
                     and ConverterUtil.get_arg(
                        op, MaceKeyword.mace_element_type_str).i
                        == EltwiseType.PROD.value) \
                    and len(op.input) == 2 \
                    and op.input[1] in self._consts \
                    and op.output_shape[0].dims[-1:] == \
                    self._consts[op.input[1]].dims \
                    and self.consumer_count(op.output[0]) == 1 \
                    and not self.is_op_output_node(op):
                consumer_op = self._consumers[op.output[0]][0]
                if (consumer_op.type == MaceOp.Eltwise.name
                    and ConverterUtil.get_arg(
                        consumer_op, MaceKeyword.mace_element_type_str).i
                        == EltwiseType.SUM.value
                    or consumer_op.type == MaceOp.BiasAdd.name) \
                        and len(consumer_op.input) == 2 \
                        and consumer_op.input[1] in self._consts \
                        and len(self._consts[consumer_op.input[1]].dims) == 1:
                    print("Fold batchnorm: %s(%s)" % (op.name, op.type))
                    consumer_op.type = MaceOp.BatchNorm.name
                    consumer_op.input[:] = [op.input[0], op.input[1],
                                            consumer_op.input[1]]
                    net.op.remove(op)
                    return True
        return False

    def fold_squared_diff_mean(self):
        net = self._model
        for op in net.op:
            if op.type == MaceOp.Eltwise.name and len(op.input) == 2:
                elt_type = ConverterUtil.get_arg(
                    op,
                    MaceKeyword.mace_element_type_str).i
                if elt_type == EltwiseType.SQR_DIFF.value and\
                        self.consumer_count(op.output[0]) == 1:
                    consumer_op = self._consumers[op.output[0]][0]
                    if consumer_op.type == MaceOp.Reduce.name:
                        axis = ConverterUtil.get_arg(
                            consumer_op,
                            MaceKeyword.mace_axis_str).ints
                        keep_dims = ConverterUtil.get_arg(
                            consumer_op,
                            MaceKeyword.mace_keepdims_str).i
                        reduce_type = ConverterUtil.get_arg(
                            consumer_op,
                            MaceKeyword.mace_reduce_type_str).i
                        if reduce_type == ReduceType.MEAN.value and\
                                len(consumer_op.input) == 1 and\
                                axis[0] == 1 and axis[1] == 2 and\
                                keep_dims > 0:
                            print("Fold SquaredDiff Reduce: %s" % op.name)
                            op.type = MaceOp.SqrDiffMean.name
                            op.output[0] = consumer_op.output[0]
                            del op.output_shape[0].dims[:]
                            op.output_shape[0].dims.extend(
                                consumer_op.output_shape[0].dims)
                            self.replace_quantize_info(op, consumer_op)
                            self.safe_remove_node(consumer_op, op)
                            return True

        return False

    def fold_moments(self):
        if self._option.device != DeviceType.HTP.value:
            return False
        net = self._model
        for op in net.op:
            if op.type == MaceOp.Reduce.name:
                axis = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_axis_str).ints
                keep_dims = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_keepdims_str).i
                reduce_type = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_reduce_type_str).i
                if reduce_type == ReduceType.MEAN.value and \
                        len(op.input) == 1 and len(axis) >= 2 and \
                        axis[0] == 1 and axis[1] == 2 and \
                        keep_dims > 0:
                    outputs = op.output
                    sqr_diff_mean_count = 0
                    for output in outputs:
                        consumer_ops = self._consumers[output]
                        for consumer_op in consumer_ops:
                            print(consumer_op.type)
                            if consumer_op.type == MaceOp.SqrDiffMean.name:
                                sqr_diff_mean_count += 1
                    if sqr_diff_mean_count == 1:
                        for output in outputs:
                            consumer_ops = self._consumers[output]
                            for consumer_op in consumer_ops:
                                if consumer_op.type == MaceOp.SqrDiffMean.name:
                                    print("Fold Moments: %s" % op.name)
                                    op.type = MaceOp.Moments.name
                                    op.output.extend(consumer_op.output)
                                    op.output_shape.extend(
                                        consumer_op.output_shape)
                                    if len(consumer_op.quantize_info) > 0:
                                        op.quantize_info.extend(
                                            consumer_op.quantize_info)
                                    net.op.remove(consumer_op)
                                    return True
        return False

    def fold_embedding_lookup(self):
        net = self._model
        for op in net.op:
            # gather -> mul
            if (op.type == MaceOp.Gather.name and
                    self.consumer_count(op.output[0]) == 1):
                consumer_op = self._consumers[op.output[0]][0]
                if (consumer_op.type == MaceOp.Eltwise.name and
                    ConverterUtil.get_arg(consumer_op,
                                          MaceKeyword.mace_element_type_str).i == EltwiseType.PROD.value and  # noqa
                            len(consumer_op.input) == 1 and
                            op.input[0] in self._consts and
                            self.consumer_count(op.input[0]) == 1):
                    print("Fold Gather and Mul: %s" % op.name)
                    gather_weights = self._consts[op.input[0]]
                    mul_weight = ConverterUtil.get_arg(consumer_op,
                                                       MaceKeyword.mace_scalar_input_str).f  # noqa
                    gather_weights.float_data[:] = [float_data * mul_weight for float_data in gather_weights.float_data]  # noqa
                    self.safe_remove_node(consumer_op, None,
                                          remove_input_tensor=True)

    def transform_lstmcell_zerostate(self):
        net = self._model

        zero_state_pattern = \
                re.compile(r'^.*BasicLSTMCellZeroState_?[0-9]*/[a-zA-Z]+_?[0-9]*')  # noqa
        for op in net.op:
            if op.type == MaceOp.Fill.name and \
                    zero_state_pattern.match(op.name):
                print("Transform lstm zerostate")
                concat_op = self._producer[op.input[0]]
                consumer_op = self._consumers[op.output[0]][0]

                if concat_op.input[0] not in self._consts or \
                        concat_op.input[1] not in self._consts:
                    continue
                dims = [self._consts[concat_op.input[0]].int32_data[0],
                        self._consts[concat_op.input[1]].int32_data[0]]
                tensor_def = net.tensors.add()
                tensor_def.name = op.output[0].replace('/zeros', '/init_const')
                tensor_def.dims.extend(dims)
                tensor_def.data_type = self._consts[op.input[1]].data_type
                tensor_def.float_data.extend(
                        [self._consts[op.input[1]].float_data[0]] *
                        (dims[0] * dims[1]))

                for i in range(len(consumer_op.input)):
                    if zero_state_pattern.match(consumer_op.input[i][:-2]):
                        consumer_op.input[i] = tensor_def.name

                net.tensors.remove(self._consts[op.input[1]])
                net.tensors.remove(self._consts[concat_op.input[0]])
                net.tensors.remove(self._consts[concat_op.input[1]])

                net.op.remove(concat_op)
                net.op.remove(op)

                return True

    def transform_basic_lstmcell(self):
        if self._option.device != DeviceType.GPU.value:
            return False

        net = self._model
        basic_lstm_concat_pattern = \
            re.compile(r'^.*basic_lstm_cell_?[0-9]*/concat_?[0-9]*')
        for op in net.op:
            if op.type == MaceOp.Concat.name and \
                    basic_lstm_concat_pattern.match(op.name):
                print("Transform basic lstmcell")
                ops_to_delete = []
                ops_to_delete.extend([op])

                op_def = net.op.add()
                op_def.name = op.name.replace('/concat', '/folded_lstmcell')
                op_def.type = MaceOp.LSTMCell.name
                op_def.arg.extend(op.arg[:-1])

                # Concat pre output and cur input
                # extend concat inputs
                op_def.input.extend([op_input for op_input in op.input])

                # lstm MatMul in FC of [pre_output, cur_input]
                matmul_op = self._consumers[op.output[0]][0]
                ops_to_delete.extend([matmul_op])
                # extend MatMul weight input
                op_def.input.extend([matmul_op.input[1]])

                # lstm BiasAdd in FC of [pre_output, cur_input]
                biasadd_op = self._consumers[matmul_op.output[0]][0]
                ops_to_delete.extend([biasadd_op])
                # extend BiasAdd bias input
                op_def.input.extend([biasadd_op.input[1]])

                # Split FC output into i, j, f, o
                # i = input_gate, j = new_input, f = forget_gate, o = output_gate  # noqa
                split_op = self._consumers[biasadd_op.output[0]][0]
                ops_to_delete.extend([split_op])

                # input gate activation
                input_gate_op = self._consumers[split_op.output[0]][0]
                ops_to_delete.extend([input_gate_op])
                # new input gate
                new_input_tanh_op = self._consumers[split_op.output[1]][0]
                ops_to_delete.extend([new_input_tanh_op])
                # forget gate add
                forget_add_op = self._consumers[split_op.output[2]][0]
                ops_to_delete.extend([forget_add_op])
                # output gate activation
                output_gate_op = self._consumers[split_op.output[3]][0]
                ops_to_delete.extend([output_gate_op])

                # extend forget add
                mace_check(len(forget_add_op.input) == 1,
                           'Wrong LSTM format in forget gate inputs')
                for arg in forget_add_op.arg:
                    if arg.name == MaceKeyword.mace_scalar_input_str:
                        op_def.arg.extend([arg])

                # state remember
                remember_mul_op = self._consumers[input_gate_op.output[0]][0]
                ops_to_delete.extend([remember_mul_op])
                mace_check(remember_mul_op.name == self._consumers[
                               new_input_tanh_op.output[0]][0].name,
                           'Wrong LSTM format in input sig & input tanh mul')

                # forget gate activation
                forget_gate_op = self._consumers[forget_add_op.output[0]][0]
                ops_to_delete.extend([forget_gate_op])

                # Mul `forget` & `pre cell state`
                forget_mul_op = self._consumers[forget_gate_op.output[0]][0]
                ops_to_delete.extend([forget_mul_op])

                # extend pre cell state input
                op_def.input.extend([forget_mul_op.input[0]])

                # get cur cell state
                # Add `forget gate output` & `remember mul output`
                remember_forget_add_op = \
                    self._consumers[remember_mul_op.output[0]][0]
                ops_to_delete.extend([remember_forget_add_op])
                mace_check(remember_forget_add_op.name ==
                           self._consumers[forget_mul_op.output[0]][0].name,
                           'Wrong LSTM format in add forget gate & remember mul')  # noqa
                op_def.output.extend([remember_forget_add_op.output[0]])
                op_def.output_shape.extend(remember_forget_add_op.output_shape)

                # cell state output tanh
                for consumer in \
                        self._consumers[remember_forget_add_op.output[0]]:
                    if consumer.type == MaceOp.Activation.name and \
                            consumer.name.find('basic_lstm_cell') > 0:
                        cell_tanh_op = consumer
                ops_to_delete.extend([cell_tanh_op])

                # final mul, get output
                final_mul_op = self._consumers[cell_tanh_op.output[0]][0]
                ops_to_delete.extend([final_mul_op])
                mace_check(final_mul_op.name ==
                           self._consumers[output_gate_op.output[0]][0].name,
                           'Wrong LSTM format in final mul')
                op_def.output.extend([final_mul_op.output[0]])
                op_def.output_shape.extend(final_mul_op.output_shape)

                for op_to_del in ops_to_delete:
                    net.op.remove(op_to_del)

                return True

        return False

    def fold_conv_and_bn(self):
        net = self._model
        for op in net.op:
            if (op.type == MaceOp.Conv2D.name) \
                    and self.consumer_count(op.output[0]) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                input_len = len(op.input)
                if (consumer_op.type == MaceOp.BatchNorm.name
                        and (input_len == 2 or (input_len == 3 and op.input[-1] in self._consts))  # noqa
                        and len(self._consumers[op.input[1]]) == 1):
                    print("Fold conv and bn: %s(%s)" % (op.name, op.type))
                    filter = self._consts[op.input[1]]
                    scale = self._consts[consumer_op.input[1]]
                    offset = self._consts[consumer_op.input[2]]
                    idx = 0
                    filter_format = self.filter_format()
                    if filter_format == DataFormat.HWIO:
                        for hwi in six.moves.range(filter.dims[0]
                                                   * filter.dims[1]
                                                   * filter.dims[2]):
                            for o in six.moves.range(filter.dims[3]):
                                filter.float_data[idx] *= scale.float_data[o]
                                idx += 1
                    elif filter_format == DataFormat.OIHW:
                        for o in six.moves.range(filter.dims[0]):
                            for hwi in six.moves.range(filter.dims[1]
                                                       * filter.dims[2]
                                                       * filter.dims[3]):
                                filter.float_data[idx] *= scale.float_data[o]
                                idx += 1
                    else:
                        mace_check(False, "filter format %s not supported" %
                                   filter_format)

                    if len(op.input) == 3:
                        conv_bias = self._consts[op.input[2]]
                        for c in six.moves.range(conv_bias.dims[0]):
                            conv_bias.float_data[c] *= scale.float_data[c]
                            conv_bias.float_data[c] += offset.float_data[c]
                        net.tensors.remove(offset)
                    else:
                        op.input.extend([consumer_op.input[2]])

                    # remove bn
                    del consumer_op.input[:]
                    net.tensors.remove(scale)
                    self.replace_quantize_info(op, consumer_op)
                    self.safe_remove_node(consumer_op, op)

                    return True

        return False

    def fold_deconv_and_bn(self):
        net = self._model
        for op in net.op:
            if (op.type in [MaceOp.Deconv2D.name, MaceOp.DepthwiseDeconv2d]) \
                    and self.consumer_count(op.output[0]) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                framework = ConverterUtil.get_arg(
                        op, MaceKeyword.mace_framework_type_str).i
                input_len = len(op.input)
                if (consumer_op.type == MaceOp.BatchNorm.name and (
                        (framework == FrameworkType.CAFFE.value and
                         (input_len == 2 or (input_len == 3 and
                                             op.input[-1] in self._consts))) or
                        (framework == FrameworkType.TENSORFLOW.value and
                         (input_len == 3 or (input_len == 4 and
                                             op.input[-1] in self._consts))))
                        and len(self._consumers[op.input[1]]) == 1):
                    print("Fold deconv and bn: %s(%s)" % (op.name, op.type))
                    filter = self._consts[op.input[1]]
                    scale = self._consts[consumer_op.input[1]]
                    offset = self._consts[consumer_op.input[2]]
                    idx = 0
                    filter_format = self.filter_format()
                    # in deconv op O and I channel is switched
                    if filter_format == DataFormat.HWIO:
                        for hw in six.moves.range(filter.dims[0]
                                                  * filter.dims[1]):
                            for o in six.moves.range(filter.dims[2]):
                                for i in six.moves.range(filter.dims[3]):
                                    filter.float_data[idx] *=\
                                        scale.float_data[o]
                                    idx += 1
                    elif filter_format == DataFormat.OIHW:
                        for i in six.moves.range(filter.dims[0]):
                            for o in six.moves.range(filter.dims[1]):
                                for hw in six.moves.range(filter.dims[2]
                                                          * filter.dims[3]):
                                    filter.float_data[idx] *=\
                                        scale.float_data[o]
                                    idx += 1
                    else:
                        mace_check(False, "filter format %s not supported" %
                                   filter_format)

                    bias_dim = -1
                    if framework == FrameworkType.CAFFE.value \
                            and len(op.input) == 3:
                        bias_dim = 2
                    if framework == FrameworkType.TENSORFLOW.value \
                            and len(op.input) == 4:
                        bias_dim = 3

                    if bias_dim != -1:
                        conv_bias = self._consts[op.input[bias_dim]]
                        for c in six.moves.range(conv_bias.dims[0]):
                            conv_bias.float_data[c] *= scale.float_data[c]
                            conv_bias.float_data[c] += offset.float_data[c]
                        net.tensors.remove(offset)
                    else:
                        op.input.extend([consumer_op.input[2]])

                    del consumer_op.input[:]
                    net.tensors.remove(scale)
                    self.replace_quantize_info(op, consumer_op)
                    self.safe_remove_node(consumer_op, op)
                    return True

        return False

    def fold_depthwise_conv_and_bn(self):
        net = self._model
        for op in net.op:
            if op.type == MaceOp.DepthwiseConv2d.name \
                    and self.consumer_count(op.output[0]) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                input_len = len(op.input)
                if (consumer_op.type == MaceOp.BatchNorm.name
                        and (input_len == 2 or (input_len == 3 and op.input[-1] in self._consts))  # noqa
                        and len(self._consumers[op.input[1]]) == 1):
                    print("Fold depthwise conv and bn: %s(%s)"
                          % (op.name, op.type))
                    filter = self._consts[op.input[1]]
                    scale = self._consts[consumer_op.input[1]]
                    offset = self._consts[consumer_op.input[2]]
                    idx = 0

                    filter_format = self.filter_format()
                    if filter_format == DataFormat.HWIO:
                        for hw in six.moves.range(filter.dims[0]
                                                  * filter.dims[1]):
                            for i in six.moves.range(filter.dims[2]):
                                for o in six.moves.range(filter.dims[3]):
                                    filter.float_data[idx] *= scale.float_data[
                                                        i * filter.dims[3] + o]
                                    idx += 1
                    elif filter_format == DataFormat.OIHW:
                        for o in six.moves.range(filter.dims[0]):
                            for i in six.moves.range(filter.dims[1]):
                                for hw in six.moves.range(filter.dims[2]
                                                          * filter.dims[3]):
                                    filter.float_data[idx] *= scale.float_data[
                                        i * filter.dims[0] + o]
                                    idx += 1
                    else:
                        mace_check(False, "filter format %s not supported" %
                                   filter_format)

                    if len(op.input) == 3:
                        conv_bias = self._consts[op.input[2]]
                        for c in six.moves.range(conv_bias.dims[0]):
                            conv_bias.float_data[c] *= scale.float_data[c]
                            conv_bias.float_data[c] += offset.float_data[c]
                        net.tensors.remove(offset)
                    else:
                        op.input.extend([consumer_op.input[2]])

                    # remove bn
                    del consumer_op.input[:]
                    net.tensors.remove(scale)
                    self.replace_quantize_info(op, consumer_op)
                    self.safe_remove_node(consumer_op, op)

                    return True

        return False

    @staticmethod
    def sort_feature_map_shape(shape, data_format):
        """Return shape in NHWC order"""
        batch = shape[0]
        if data_format == DataFormat.NHWC:
            height = shape[1]
            width = shape[2]
            channels = shape[3]
        else:
            height = shape[2]
            width = shape[3]
            channels = shape[1]
        return batch, height, width, channels

    @staticmethod
    def sort_filter_shape(filter_shape, filter_format):
        """Return filter shape in HWIO order"""
        if filter_format == DataFormat.HWIO:
            filter_height = filter_shape[0]
            filter_width = filter_shape[1]
            in_channels = filter_shape[2]
            out_channels = filter_shape[3]
        elif filter_format == DataFormat.OIHW:
            filter_height = filter_shape[2]
            filter_width = filter_shape[3]
            in_channels = filter_shape[1]
            out_channels = filter_shape[0]
        elif filter_format == DataFormat.HWOI:
            filter_height = filter_shape[0]
            filter_width = filter_shape[1]
            in_channels = filter_shape[3]
            out_channels = filter_shape[2]
        else:
            mace_check(False, "filter format %s not supported" % filter_format)
        return filter_height, filter_width, in_channels, out_channels

    def transform_add_to_biasadd(self):
        if self._option.device == DeviceType.HTP.value:
            return False
        net = self._model
        for op in net.op:
            if (not self._has_none_df and op.type == 'Eltwise'
                    and ConverterUtil.get_arg(op, MaceKeyword.mace_element_type_str).i == EltwiseType.SUM.value  # noqa
                    and len(op.input) == 2
                    and op.input[1] in self._consts
                    and len(self._consts[op.input[1]].dims) == 1):
                print("Transform add to biasadd: %s(%s)" % (op.name, op.type))
                op.type = MaceOp.BiasAdd.name
                return True

        return False

    def replace_quantize_info(self, op, replace_op):
        if len(replace_op.quantize_info) > 0:
            del op.quantize_info[:]
            op.quantize_info.extend(replace_op.quantize_info)
            for i in range(len(op.quantize_info)):
                self._quantize_activation_info[op.output[i]] = \
                    op.quantize_info[i]

    def fold_biasadd(self):
        net = self._model
        for op in net.op:
            framework = ConverterUtil.get_arg(op, MaceKeyword.mace_framework_type_str).i
            if (((op.type == MaceOp.Conv2D.name
                  or op.type == MaceOp.DepthwiseConv2d.name
                  or op.type == MaceOp.FullyConnected.name)
                 and len(op.input) == 2)
                or (op.type == MaceOp.Deconv2D.name
                    and ((framework == FrameworkType.CAFFE.value
                          and len(op.input) == 2)
                         or (framework == FrameworkType.TENSORFLOW.value
                             and len(op.input) == 3)))) \
                    and len(self._consumers.get(op.output[0], [])) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                is_add = False
                if consumer_op.type == MaceOp.Eltwise.name and \
                        self._option.device == DeviceType.HTP.value:
                    is_add = (ConverterUtil.get_arg(consumer_op,
                                                    MaceKeyword.mace_element_type_str).i ==
                              EltwiseType.SUM.value)
                if consumer_op.type == MaceOp.BiasAdd.name or is_add:
                    print("Fold biasadd: %s(%s)" % (op.name, op.type))
                    op.name = consumer_op.name
                    op.output[0] = consumer_op.output[0]
                    op.input.append(consumer_op.input[1])
                    self.replace_quantize_info(op, consumer_op)
                    self.safe_remove_node(consumer_op, op)
                    return True

        return False

    def flatten_atrous_conv(self):
        if self._option.device != DeviceType.GPU.value \
               and self._option.device != DeviceType.APU.value \
               and self._option.device != DeviceType.HTA.value \
               and self._option.device != DeviceType.HTP.value:
            return

        net = self._model
        for op in net.op:
            if (op.type == MaceOp.SpaceToBatchND.name
                    and len(self._consumers.get(op.output[0], [])) == 1):
                conv_op = self._consumers.get(op.output[0])[0]
                if (conv_op.type == MaceOp.Conv2D.name
                        or conv_op.type == MaceOp.DepthwiseConv2d.name) \
                        and len(self._consumers.get(conv_op.output[0], [])) == 1:  # noqa
                    b2s_op = self._consumers.get(conv_op.output[0])[0]
                    if b2s_op.type == MaceOp.BatchToSpaceND.name:
                        six.print_("Flatten atrous convolution")
                        # Add args.
                        padding_arg_values = ConverterUtil.get_arg(
                            op,
                            MaceKeyword.mace_paddings_str).ints
                        blocks_arg_values = ConverterUtil.get_arg(
                            b2s_op,
                            MaceKeyword.mace_space_batch_block_shape_str).ints
                        dilation_arg = ConverterUtil.get_arg(
                            conv_op,
                            MaceKeyword.mace_dilations_str)
                        if dilation_arg is None:
                            dilation_arg = conv_op.arg.add()
                        dilation_arg.name = MaceKeyword.mace_dilations_str
                        dilation_arg.ints[:] = blocks_arg_values

                        padding_arg = ConverterUtil.get_arg(
                            conv_op,
                            MaceKeyword.mace_padding_str)
                        if padding_arg is None:
                            padding_arg = conv_op.arg.add()
                        padding_arg.name = MaceKeyword.mace_padding_str
                        if len(padding_arg_values) > 0 \
                                and padding_arg_values[0] > 0:
                            padding_arg.i = PaddingMode.SAME.value

                        strides_arg = ConverterUtil.get_arg(
                            conv_op,
                            MaceKeyword.mace_strides_str)
                        if strides_arg is None:
                            strides_arg = conv_op.arg.add()
                        strides_arg.name = MaceKeyword.mace_strides_str
                        strides_arg.ints[:] = [1, 1]

                        # update output shape
                        conv_op.output_shape[0].dims[:] = \
                            b2s_op.output_shape[0].dims[:]
                        conv_op.output[0] = b2s_op.output[0]
                        conv_op.name = b2s_op.name

                        self.safe_remove_node(op, None)
                        self.replace_quantize_info(b2s_op, conv_op)
                        self.safe_remove_node(b2s_op, conv_op)
                        return True
        return False

    def fold_activation(self):
        if self._option.device == DeviceType.HTP.value:
            return
        net = self._model
        for op in net.op:
            if (op.type == MaceOp.Conv2D.name
                or op.type == MaceOp.Deconv2D.name
                or op.type == MaceOp.DepthwiseConv2d.name
                or op.type == MaceOp.FullyConnected.name
                or op.type == MaceOp.BatchNorm.name
                or op.type == MaceOp.InstanceNorm.name) \
                    and len(self._consumers.get(op.output[0], [])) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                fold_consumer = False
                if consumer_op.type == MaceOp.Activation.name:
                    act_type_arg = ConverterUtil.get_arg(
                        consumer_op, MaceKeyword.mace_activation_type_str)
                    act_type = act_type_arg.s.decode()
                    if self._option.device == DeviceType.APU.value:
                        fold_consumer = (act_type in
                                         [ActivationType.RELU.name,
                                          ActivationType.RELUX.name])
                    else:
                        fold_consumer = (act_type != ActivationType.PRELU.name)
                    # during quantization, only fold relu/relux
                    if (self._option.quantize_stat or self._option.quantize) \
                            and act_type not in [ActivationType.RELU.name,
                                                 ActivationType.RELUX.name]:
                        continue
                if fold_consumer:
                    print("Fold activation: %s(%s)" % (op.name, op.type))
                    op.name = consumer_op.name
                    op.output[0] = consumer_op.output[0]
                    for arg in consumer_op.arg:
                        if arg.name == MaceKeyword.mace_activation_type_str \
                                or arg.name == MaceKeyword.mace_activation_max_limit_str \
                                or arg.name == MaceKeyword.mace_activation_coefficient_str \
                                or arg.name == MaceKeyword.mace_hardsigmoid_alpha_str \
                                or arg.name == MaceKeyword.mace_hardsigmoid_beta_str:
                            op.arg.extend([arg])

                    self.replace_quantize_info(op, consumer_op)
                    self.safe_remove_node(consumer_op, op)
                    return True

        return False

    def transform_global_conv_to_fc(self):
        """Transform global conv to fc should be placed after transposing
        input/output and filter"""

        net = self._model
        for op in net.op:
            if op.type == MaceOp.Conv2D.name \
                    and len(op.input) >= 2 \
                    and op.input[1] in self._consts and \
                    op.input[0] in self._producer:
                producer = self._producer[op.input[0]]
                input_shape = producer.output_shape[0].dims
                batch, height, width, channels = self.sort_feature_map_shape(
                    input_shape, ConverterUtil.data_format(producer))
                filter = self._consts[op.input[1]]
                filter_shape = filter.dims
                filter_height, filter_width, in_channels, out_channels = \
                    self.sort_filter_shape(filter_shape, self.filter_format())
                zero_padding = True
                padding_arg = ConverterUtil.get_arg(op,
                                                    MaceKeyword.mace_padding_str)  # noqa
                if padding_arg is not None:
                    if padding_arg.i != PaddingMode.VALID.value:
                        zero_padding = False
                else:
                    padding_value_arg = ConverterUtil.get_arg(op,
                                                              MaceKeyword.mace_padding_values_str)  # noqa
                    if padding_value_arg is not None:
                        if not all(v == 0 for v in padding_value_arg.ints):
                            zero_padding = False

                if height == filter_height and width == filter_width \
                        and zero_padding \
                        and len(self._consumers[op.input[1]]) == 1:
                    print("transform global conv to fc %s(%s)"
                          % (op.name, op.type))
                    op.type = MaceOp.FullyConnected.name

        return False

    def reshape_fc_weight(self):
        if self._option.device in [DeviceType.APU.value, DeviceType.HTP.value]:
            return
        net = self._model
        filter_format = self.filter_format()
        for op in net.op:
            if op.type == MaceOp.FullyConnected.name:
                weight = self._consts[op.input[1]]
                if len(weight.dims) == 2:
                    print("Reshape fully connected weight shape")
                    input_op = self._producer[op.input[0]]
                    input_shape = list(input_op.output_shape[0].dims)
                    weight.dims[:] = [weight.dims[0]] + input_shape[1:]
                    if len(input_shape) == 2:
                        if filter_format == DataFormat.HWIO:
                            weight.dims[:] = [1, 1] + weight.dims[:]
                        elif filter_format == DataFormat.OIHW:
                            weight.dims[:] = weight.dims[:] + [1, 1]
                        else:
                            mace_check(False,
                                       "FC does not support filter format %s" %
                                       filter_format.name)
        return False

    def add_winograd_arg(self):
        if self._wino_arg == 0:
            return False
        net = self._model

        for op in net.op:
            if op.type == MaceOp.Conv2D.name:
                winograd_arg = op.arg.add()
                winograd_arg.name = MaceKeyword.mace_wino_arg_str
                winograd_arg.i = self._wino_arg

        return False

    def transpose_matmul_weight(self):
        if self._option.device != DeviceType.CPU.value:
            return False
        net = self._model
        transposed_weights = []
        for op in net.op:
            if op.type == MaceOp.MatMul.name:  # noqa
                rhs = op.input[1]
                if rhs in self._consts and len(self._consts[rhs].dims) == 2:
                    arg = ConverterUtil.get_arg(op, MaceKeyword.mace_transpose_b_str)  # noqa
                    # six.print_("Transpose matmul weight %s" % rhs)
                    if arg is None:
                        arg = op.arg.add()
                        arg.name = MaceKeyword.mace_transpose_b_str
                        arg.i = 0
                    if arg.i == 0:
                        arg.i = 1
                        if rhs not in transposed_weights:
                            filter = self._consts[rhs]
                            filter_data = np.array(filter.float_data).reshape(
                                filter.dims)
                            filter_data = filter_data.transpose(1, 0)
                            filter.float_data[:] = filter_data.flat
                            filter.dims[:] = filter_data.shape
                            transposed_weights.append(rhs)
                            six.print_('Transpose matmul weight to shape:',
                                       filter.dims)

    def transpose_filters(self):
        net = self._model
        filter_format = self.filter_format()
        transposed_filter = set()
        transposed_deconv_filter = set()

        if (((self._option.quantize and
                self._option.device == DeviceType.CPU.value) or
                self._option.device == DeviceType.APU.value) and
                (not self._option.quantize_schema == MaceKeyword.mace_int8)):
            print("Transpose filters to OHWI")
            if filter_format == DataFormat.HWIO:
                transpose_order = [3, 0, 1, 2]
            elif filter_format == DataFormat.OIHW:
                transpose_order = [0, 2, 3, 1]
            else:
                mace_check(False, "Quantize model does not support conv "
                           "filter format: %s" % filter_format.name)

            for op in net.op:
                if (op.type == MaceOp.Conv2D.name or
                    op.type == MaceOp.Deconv2D.name or
                    (op.type == MaceOp.DepthwiseConv2d.name and
                     self._option.device == DeviceType.APU.value) or
                    (op.type == MaceOp.FullyConnected.name and
                     len(self._consts[op.input[1]].dims) == 4)) and \
                        op.input[1] not in transposed_filter:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(transpose_order)
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                    transposed_filter.add(op.input[1])
                elif op.type == MaceOp.DepthwiseConv2d.name and\
                        filter_format == DataFormat.OIHW and\
                        self._option.device == DeviceType.CPU.value and\
                        op.input[1] not in transposed_filter:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(2, 3, 1, 0)
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                    transposed_filter.add(op.input[1])
            # deconv's filter's output channel and input channel is reversed
            for op in net.op:
                if op.type == MaceOp.Deconv2D.name and \
                        op.input[1] not in transposed_deconv_filter:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(3, 1, 2, 0)
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                    transposed_deconv_filter.add(op.input[1])

            self.set_filter_format(DataFormat.OHWI)
        elif (self._option.device == DeviceType.HEXAGON.value or
              self._option.device == DeviceType.HTA.value or
              self._option.device == DeviceType.HTP.value):
            print("Transpose filters to HWIO/HWIM")
            for op in net.op:
                has_data_format = ConverterUtil.data_format(op) == \
                                DataFormat.AUTO
                if has_data_format and (op.type == MaceOp.Eltwise.name or
                                        op.type == MaceOp.Concat.name):
                    for i in range(len(op.input)):
                        if op.input[i] in self._consts and \
                                op.input[i] not in transposed_filter:
                            filter = self._consts[op.input[i]]
                            filter_data = np.array(filter.float_data).reshape(
                                filter.dims)
                            if len(filter_data.shape) == 1 and \
                                    len(op.output_shape[0].dims) == 4:
                                filter_data = np.array(filter.float_data).reshape(
                                    [1, 1, 1, filter.dims[0]])
                            if filter_format == DataFormat.OIHW:
                                filter_data = filter_data.transpose(0, 2, 3, 1)
                            else:
                                print(op.type, op.name, filter_format)
                                mace_check(False, "Unsupported filter format.")
                            filter.float_data[:] = filter_data.flat
                            filter.dims[:] = filter_data.shape
                            transposed_filter.add(op.input[i])
                if filter_format == DataFormat.OIHW and \
                        (op.type == MaceOp.Conv2D.name or
                         (op.type == MaceOp.DepthwiseConv2d.name and
                          (self._option.device == DeviceType.HEXAGON.value or
                           self._option.device == DeviceType.HTA.value)) or
                         (op.type == MaceOp.FullyConnected.name and
                          len(self._consts[op.input[1]].dims) == 4)) and \
                        op.input[1] in self._consts and \
                        op.input[1] not in transposed_filter:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(2, 3, 1, 0)
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                    transposed_filter.add(op.input[1])

                if (op.type == MaceOp.Deconv2D.name or
                        op.type == MaceOp.DepthwiseDeconv2d.name) \
                        and op.input[1] in self._consts \
                        and op.input[1] not in transposed_deconv_filter:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    if filter_format == DataFormat.HWIO:
                        # from HWOI to OHWI
                        filter_data = filter_data.transpose(2, 0, 1, 3)
                    elif filter_format == DataFormat.OIHW:
                        if self._option.device == DeviceType.HTP.value:
                            filter_data = filter_data.transpose(2, 3, 0, 1)
                        else:
                            # from IOHW to OHWI
                            filter_data = filter_data.transpose(1, 2, 3, 0)
                    else:
                        mace_check(False, "Unsupported filter format.")
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                    transposed_deconv_filter.add(op.input[1])

                if op.type == MaceOp.DepthwiseConv2d.name \
                        and self._option.device == DeviceType.HTP.value:
                    filter = self._consts[op.input[1]]
                    dims = filter.dims[:]
                    if filter_format == DataFormat.HWIO:
                        filter.dims[:] = \
                            [dims[0], dims[1], 1, dims[2] * dims[3]]
                    elif filter_format == DataFormat.OIHW:
                        # from MIHW to HW1O
                        filter_data = np.array(filter.float_data).reshape(dims)
                        filter_data = filter_data.transpose(2, 3, 0, 1)
                        filter.float_data[:] = filter_data.flat
                        filter.dims[:] = filter_data.shape
                    else:
                        mace_check(False, "Unsupported filter format.")
                    transposed_filter.add(op.input[1])
        else:
            # transpose filter to OIHW/MIHW for tensorflow (HWIO/HWIM)
            if filter_format == DataFormat.HWIO:
                for op in net.op:
                    if (op.type == MaceOp.Conv2D.name
                            or op.type == MaceOp.Deconv2D.name
                            or op.type == MaceOp.DepthwiseConv2d.name) \
                            and op.input[1] in self._consts \
                            and op.input[1] not in transposed_filter:
                        print("Transpose Conv2D/Deconv2D filters to OIHW/MIHW")
                        filter = self._consts[op.input[1]]
                        filter_data = np.array(filter.float_data).reshape(
                            filter.dims)
                        filter_data = filter_data.transpose(3, 2, 0, 1)
                        filter.float_data[:] = filter_data.flat
                        filter.dims[:] = filter_data.shape
                        transposed_filter.add(op.input[1])
                    if (op.type == MaceOp.MatMul.name and
                            (ConverterUtil.get_arg(
                                op,
                                MaceKeyword.mace_winograd_filter_transformed)
                                 is not None)  # noqa
                            and op.input[1] not in transposed_filter):
                        print("Transpose Winograd filters to OIHW/MIHW")
                        filter = self._consts[op.input[0]]
                        filter_data = np.array(filter.float_data).reshape(
                            filter.dims)
                        filter_data = filter_data.transpose(3, 2, 0, 1)
                        filter.float_data[:] = filter_data.flat
                        filter.dims[:] = filter_data.shape
                        transposed_filter.add(op.input[0])
                    if op.type == MaceOp.FullyConnected.name \
                            and op.input[1] not in transposed_filter:
                        weight = self._consts[op.input[1]]
                        if len(weight.dims) == 4:
                            print("Transpose FullyConnected filters to"
                                  " OIHW/MIHW")
                            weight_data = np.array(weight.float_data).reshape(
                                weight.dims)
                            weight_data = weight_data.transpose(3, 2, 0, 1)
                            weight.float_data[:] = weight_data.flat
                            weight.dims[:] = weight_data.shape
                            transposed_filter.add(op.input[1])

                self.set_filter_format(DataFormat.OIHW)
            # deconv's filter's output channel and input channel is reversed
            for op in net.op:
                if op.type in [MaceOp.Deconv2D.name,
                               MaceOp.DepthwiseDeconv2d] \
                        and op.input[1] in self._consts \
                        and op.input[1] not in transposed_deconv_filter:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(1, 0, 2, 3)
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                    transposed_deconv_filter.add(op.input[1])

        return False

    def fold_reshape(self):
        net = self._model
        for op in net.op:
            if op.type == MaceOp.Softmax.name:
                # see if possible to fold
                # Reshape(xd->2d) + Softmax(2d) [+ Reshape(xd)] to Softmax(xd)
                should_fold = False
                if op.input[0] in self._producer \
                        and self._producer[op.input[0]].type \
                        == MaceOp.Reshape.name \
                        and len(op.output_shape[0].dims) == 2:
                    producer = self._producer[op.input[0]]
                    reshape_input_rank = len(self.get_tensor_shape(
                        producer.input[0]))
                    if reshape_input_rank == 4:
                        should_fold = True

                if should_fold:
                    print(
                        "Fold reshape and softmax: %s(%s)"
                        % (op.name, op.type))
                    producer = self._producer[op.input[0]]
                    op.output_shape[0].dims[:] = self.get_tensor_shape(
                        producer.input[0])

                    if op.output[0] in self._consumers:
                        consumer = self._consumers[op.output[0]][0]
                        # if there is a shape op, remove it too
                        if len(consumer.input) > 1:
                            if (consumer.input[1] in self._producer
                                and self._producer[consumer.input[1]].type
                                    == 'Shape'):
                                self.safe_remove_node(
                                    self._producer[consumer.input[1]], None,
                                    remove_input_tensor=True)
                        # remove consumer reshape
                        self.safe_remove_node(consumer, op,
                                              remove_input_tensor=True)
                    # remove producer reshape
                    self.safe_remove_node(producer,
                                          self._producer.get(producer.input[0],
                                                             None),
                                          remove_input_tensor=True)

                    return True
        return False

    def is_after_fc(self, op):
        while op.input[0] in self._producer:
            producer = self._producer[op.input[0]]
            if producer.type in [MaceOp.Activation.name, MaceOp.BiasAdd.name]:
                op = producer
                continue
            elif producer.type == MaceOp.FullyConnected.name:
                return True
            else:
                return False
        return False

    def transform_matmul_to_fc(self):
        net = self._model
        filter_format = self.filter_format()
        for op in net.op:
            # transform `matmul` to `fc(2D)` for APU and HTP
            is_htp = self._option.device == DeviceType.HTP.value
            if self._option.device == DeviceType.APU.value or is_htp:
                if op.type == MaceOp.MatMul.name:
                    transpose_a_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_transpose_a_str)  # noqa
                    transpose_b_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_transpose_b_str)  # noqa
                    transpose_a = transpose_a_arg is not None and transpose_a_arg.i == 1  # noqa
                    transpose_b = transpose_b_arg is not None and transpose_b_arg.i == 1  # noqa
                    if transpose_a is False and transpose_b is False and \
                            op.input[1] in self._consts and \
                            len(self.get_tensor_shape(op.input[0])) == 2 and \
                            len(self.get_tensor_shape(op.input[1])) == 2:
                        # transform `reshape(2D) -> matmul` to `fc(2D)` for HTP
                        if op.input[0] in self._producer:
                            product_op = self._producer[op.input[0]]
                            if is_htp and product_op.type == MaceOp.Reshape.name:
                                consumers = self._consumers[product_op.output[0]]
                                print('convert reshape and matmul to fc')
                                self.safe_remove_node(product_op, None,
                                                      remove_input_tensor=True)
                                for matmul_op in consumers:
                                    matmul_op.type = MaceOp.FullyConnected.name
                                    filter = self._consts[matmul_op.input[1]]
                                    filter_data = \
                                        np.array(filter.float_data).reshape(filter.dims)
                                    filter_data = filter_data.transpose(1, 0)
                                    filter.float_data[:] = filter_data.flat
                                    filter.dims[:] = filter_data.shape
                                    six.print_('Transpose matmul weight to shape:',
                                               filter.dims)
                                return True
                        op.type = MaceOp.FullyConnected.name
                        filter = self._consts[op.input[1]]
                        filter_data = \
                            np.array(filter.float_data).reshape(filter.dims)
                        filter_data = filter_data.transpose(1, 0)
                        filter.float_data[:] = filter_data.flat
                        filter.dims[:] = filter_data.shape
                        six.print_('Transpose matmul weight to shape:',
                                   filter.dims)
                        return True
                continue
            # transform `input(4D) -> reshape(2D) -> matmul` to `fc(2D)`
            # fc output is 2D in transformer, using as 4D in op kernel
            # work for TensorFlow/PyTorch/ONNX
            framework = ConverterUtil.framework_type(net)
            is_torch = framework == FrameworkType.PYTORCH.value
            is_tf = framework == FrameworkType.TENSORFLOW.value
            is_onnx = framework == FrameworkType.ONNX.value

            if is_htp and op.type == MaceOp.Reshape.name and \
                    len(op.input) == 2 and \
                    op.input[1] in self._consts and \
                    len(op.output_shape[0].dims) == 2 and \
                    (is_tf or is_torch or is_onnx) and \
                    op.input[0] in self._producer and \
                    op.output[0] in self._consumers:
                input_op = self._producer[op.input[0]]
                input_shape = input_op.output_shape[0].dims
                # check input op
                if len(input_shape) == 4 and \
                        np.prod(input_shape[1:]) == op.output_shape[0].dims[1]:
                    is_fc = True
                    consumers = self._consumers[op.output[0]]
                    # check matmul op
                    for matmul_op in consumers:
                        if matmul_op.type != MaceOp.MatMul.name:
                            is_fc = False
                        else:
                            weight = self._consts[matmul_op.input[1]]
                            od = op.output_shape[0].dims
                            wd = weight.dims
                            if len(wd) != 2:
                                is_fc = False
                            # tf fc weight: IO; onnx/pytorch fc weight: OI
                            if (is_tf and wd[0] != od[1]) or \
                                    ((is_torch or is_onnx) and wd[1] != od[1]):
                                is_fc = False
                    if is_fc:
                        print('convert reshape and matmul to fc')
                        self.safe_remove_node(op, None,
                                              remove_input_tensor=True)
                        for matmul_op in consumers:
                            weight = self._consts[matmul_op.input[1]]
                            matmul_op.type = MaceOp.FullyConnected.name
                            weight_data = np.array(weight.float_data).reshape(
                                weight.dims)
                            if is_tf:
                                weight.dims[:] = input_shape[1:] + \
                                    [weight_data.shape[1]]
                            if is_torch or is_onnx:
                                in_data_format = ConverterUtil.data_format(
                                    input_op)
                                # OI+NCHW[2:]=OIHW
                                if in_data_format == DataFormat.NCHW:
                                    size = input_shape[2] * input_shape[3]
                                    mace_check(weight.dims[1] % size == 0,
                                               "Reshape dims of input cannot be \
                                                divisible by dims of output")
                                    weight.dims[1] = weight.dims[1] // size
                                    weight.dims.extend(input_shape[2:])
                                # OI+NHWC[1:2]=OIHW
                                else:
                                    size = input_shape[1] * input_shape[2]
                                    mace_check(weight.dims[1] % size == 0,
                                               "Reshape dims of input cannot be \
                                                divisible by dims of output")
                                    weight.dims[1] = weight.dims[1] // size
                                    weight.dims.extend(input_shape[1:2])
                        return True

            # transform `fc1(2D) -> matmul` to `fc1(2D) -> fc1(2D)`
            if op.type == MaceOp.MatMul.name and \
                    (is_tf or is_torch or is_onnx) and \
                    op.input[1] in self._consts:
                producer = self._producer[op.input[0]]
                weight = self._consts[op.input[1]]
                if len(weight.dims) == 2 and self.is_after_fc(op) and \
                        len(producer.output_shape[0].dims) == 2 and \
                        ((is_tf and weight.dims[0] == producer.output_shape[0].dims[1]) or  # noqa
                         (is_torch and weight.dims[1] == producer.output_shape[0].dims[1]) or  # noqa
                         (is_onnx and weight.dims[1] == producer.output_shape[0].dims[1])):  # noqa
                    six.print_('convert matmul to fc')
                    op.type = MaceOp.FullyConnected.name
                    weight_data = np.array(weight.float_data).reshape(
                        weight.dims)
                    # only 1 of the 2 branches can be executed
                    if is_tf:
                        weight.dims[:] = [1, 1] + list(weight_data.shape)
                    if is_torch or is_onnx:
                        weight.dims.extend([1, 1])
                    return True
        return False

    # When FC is producer of Reshape, updating output shape of FC makes
    # Reshape transposable.
    def update_fc_output_shape(self):
        net = self._model
        framework = ConverterUtil.framework_type(net)
        is_torch = framework == FrameworkType.PYTORCH.value
        is_onnx = framework == FrameworkType.ONNX.value
        dev = self._option.device
        if not ((is_torch or is_onnx) and
                (dev == DeviceType.GPU.value or dev == DeviceType.CPU.value)):
            return False
        for op in net.op:
            if op.type != MaceOp.FullyConnected.name:
                continue
            out_data_format = ConverterUtil.data_format(op)
            if len(op.output_shape[0].dims) != 2:
                continue
            if out_data_format == DataFormat.NCHW:
                op.output_shape[0].dims.extend([1, 1])
            else:
                dim1 = op.output_shape[0].dims[1]
                del op.output_shape[0].dims[1:]
                op.output_shape[0].dims.extend([1, 1, dim1])
        return False

    def update_float_op_data_type(self):
        print("update op with float data type")
        net = self._model
        data_type = self._option.data_type
        net.data_type = data_type

        if self._option.quantize:
            return

        for op in net.op:
            data_type_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_op_data_type_str)
            if not data_type_arg:
                data_type_arg = op.arg.add()
                data_type_arg.name = MaceKeyword.mace_op_data_type_str
                data_type_arg.i = data_type
            elif data_type_arg.i != data_type \
                    and data_type_arg.i == mace_pb2.DT_FLOAT:
                data_type_arg.i = data_type

        return False

    def sort_dfs(self, op, visited, sorted_nodes):
        if op.name in visited:
            return
        visited.update([op.name])
        if len(op.input) > 0:
            for input_tensor in op.input:
                producer_op = self._producer.get(input_tensor, None)
                if producer_op is None:
                    pass
                elif producer_op.name not in visited:
                    self.sort_dfs(producer_op, visited, sorted_nodes)
        sorted_nodes.append(op)

    def sort_by_execution(self):
        print("Sort by execution")
        net = self._model
        visited = set()
        sorted_nodes = []

        output_nodes = list(self._option.check_nodes.keys())
        if not self._quantize_activation_info:
            output_nodes.extend(self._option.output_nodes)
        for output_node in output_nodes:
            mace_check(output_node in self._producer,
                       "output_tensor %s not existed in model" % output_node)
            self.sort_dfs(self._producer[output_node], visited, sorted_nodes)

        del net.op[:]
        net.op.extend(sorted_nodes)

        print("Final ops:")
        index = 0
        for op in net.op:
            if op.type not in [MaceOp.Quantize.name, MaceOp.Dequantize.name]:
                index_str = str(index)
                index += 1
            else:
                index_str = ''
            print("%s (%s, index:%s): %s" % (op.name, op.type, index_str, [
                out_shape.dims for out_shape in op.output_shape]))
        return False

    def is_transposable_data_format_ops(self, op):
        transposable = op.type in MaceTransposableDataFormatOps
        framework = ConverterUtil.framework_type(self._model)
        is_torch = framework == FrameworkType.PYTORCH.value
        is_onnx = framework == FrameworkType.ONNX.value

        if op.type == MaceOp.Reshape:
            input_op = self._producer[op.input[0]]
            if len(input_op.output_shape) == 0 or len(op.output_shape) == 0:
                transposable = False
            else:
                input_dims = input_op.output_shape[0].dims
                output_dims = op.output_shape[0].dims
                if len(input_op.output_shape) != 1 or \
                        len(input_dims) != 4 or len(output_dims) != 4:
                    transposable = False
                else:
                    if is_torch or is_onnx:
                        transposable = True
                    else:
                        in_b, in_h, in_w, in_c = self.sort_feature_map_shape(
                            input_dims, ConverterUtil.data_format(input_op))
                        ou_b, ou_h, ou_w, ou_c = self.sort_feature_map_shape(
                            output_dims, ConverterUtil.data_format(op))
                        transposable = (in_b == ou_b and in_c == ou_c)
                        if self._option.device == DeviceType.HEXAGON.value or \
                                self._option.device == DeviceType.HTP.value:
                            transposable = True
        elif op.type == MaceOp.Squeeze:
            input_op = self._producer[op.input[0]]
            if len(input_op.output_shape) == 0 or len(op.output_shape) == 0:
                transposable = False
            else:
                input_dims = input_op.output_shape[0].dims
                output_dims = op.output_shape[0].dims
                src_df = ConverterUtil.data_format(self._model)
                arg = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str)
                if len(input_dims) == 4 and len(output_dims) == 2 and \
                        ((src_df == DataFormat.NCHW and arg.ints == [2, 3]) or
                         (src_df == DataFormat.NHWC and arg.ints == [1, 2])):
                    transposable = True
                else:
                    transposable = False
        elif op.type == MaceOp.Transpose:
            if op.output[0] in self._consumers:
                consumer = self._consumers[op.output[0]][0]
                if consumer.type == MaceOp.Reshape:
                    transposable = False
        elif op.type == MaceOp.FullyConnected and \
                self._option.device == DeviceType.HTP.value:
            transposable = False

        if op.type in MaceTransposableDataFormatOps and not transposable:
            print("%s(%s) is not a transposable op in this model."
                  % (op.name, op.type))
        return transposable

    def update_data_format(self):
        print("update data format")
        net = self._model
        for op in net.op:
            df_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_data_format_str)
            if not df_arg:
                df_arg = op.arg.add()
                df_arg.name = MaceKeyword.mace_data_format_str
            if op.type in MaceFixedDataFormatOps:
                df_arg.i = DataFormat.AUTO.value
            elif self.is_transposable_data_format_ops(op):
                input_df = DataFormat.AUTO.value
                for input_tensor in op.input:
                    if input_tensor in self._consts:
                        continue
                    mace_check(
                        input_tensor in self._producer,
                        "Input tensor %s not in producer" % input_tensor)
                    father_op = self._producer[input_tensor]
                    temp_input_df = ConverterUtil.get_arg(
                        father_op, MaceKeyword.mace_data_format_str)
                    if temp_input_df.i != DataFormat.AUTO.value:
                        input_df = temp_input_df.i
                if input_df == DataFormat.AUTO.value:
                    df_arg.i = input_df
                    # add flag to mark the ops may has data format
                    has_data_format_arg = op.arg.add()
                    has_data_format_arg.name = \
                        MaceKeyword.mace_has_data_format_str
                    has_data_format_arg.i = 1
        return False

    def transpose_data_format(self):
        print("Transpose arguments based on data format")
        net = self._model

        src_data_format = ConverterUtil.data_format(net)
        for op in net.op:
            has_data_format = ConverterUtil.data_format(op) == \
                              DataFormat.AUTO
            # transpose args
            if op.type == MaceOp.Pad.name:
                for arg in op.arg:
                    if arg.name == MaceKeyword.mace_paddings_str:
                        mace_check(len(arg.ints) == 8,
                                   "pad dim rank should be 8.")
                        if src_data_format == DataFormat.NCHW and \
                                has_data_format:
                            print("Transpose pad args: %s(%s)"
                                  % (op.name, op.type))
                            self.transpose_shape(arg.ints,
                                                 [0, 1, 4, 5, 6, 7, 2, 3])
            elif op.type == MaceOp.Concat.name or op.type == MaceOp.Split.name:
                for arg in op.arg:
                    if arg.name == MaceKeyword.mace_axis_str:
                        if (src_data_format == DataFormat.NCHW
                                and has_data_format
                                and len(op.output_shape[0].dims) == 4):
                            print("Transpose concat/split args: %s(%s)"
                                  % (op.name, op.type))
                            if arg.i < 0:
                                arg.i += 4
                            if arg.i == 1:
                                arg.i = 3
                            elif arg.i == 2:
                                arg.i = 1
                            elif arg.i == 3:
                                arg.i = 2
                        if op.input[0] in self._producer:
                            producer = self._producer[op.input[0]]
                            input_shape = producer.output_shape[0].dims
                            if (producer.type == MaceOp.FullyConnected.name
                                    and len(input_shape) == 2):
                                axis_arg = ConverterUtil.get_arg(
                                        op, MaceKeyword.mace_axis_str)
                                if axis_arg.i == 1:
                                    axis_arg.i = 3

            elif op.type == MaceOp.Squeeze.name:
                for arg in op.arg:
                    if arg.name == MaceKeyword.mace_axis_str:
                        if (src_data_format == DataFormat.NCHW
                                and has_data_format
                                and len(self._producer[op.input[0]].output_shape[0].dims) == 4  # noqa
                                and len(op.output_shape[0].dims) == 2
                                and arg.ints == [2, 3]):
                            print("Transpose squeeze args: %s(%s)"
                                  % (op.name, op.type))
                            arg.ints[:] = [1, 2]

            elif op.type == MaceOp.Reduce.name:
                for arg in op.arg:
                    if arg.name == MaceKeyword.mace_axis_str:
                        if src_data_format == DataFormat.NCHW and \
                                has_data_format:
                            print("Transpose reduce args: %s(%s)"
                                  % (op.name, op.type))
                            reduce_axises = list(arg.ints)
                            new_axises = []
                            for i in range(len(reduce_axises)):
                                idx = reduce_axises[i]
                                if idx == 2 or idx == 3:
                                    new_axises.append(idx - 1)
                                elif idx == 1:
                                    new_axises.append(3)
                                elif idx == -1:
                                    new_axises.append(2)
                                else:
                                    new_axises.append(idx)
                            new_axises.sort()
                            arg.ints[:] = []
                            arg.ints.extend(new_axises)
            elif op.type == MaceOp.Crop.name:
                offset_arg = ConverterUtil.get_arg(op,
                                                   MaceKeyword.mace_offset_str)
                mace_check(offset_arg and
                           src_data_format == DataFormat.NCHW
                           and has_data_format
                           and len(op.output_shape[0].dims) == 4,
                           "MACE only support crop with NCHW format")
                print("Transpose crop args: %s(%s)"
                      % (op.name, op.type))
                self.transpose_shape(offset_arg.ints, [0, 2, 3, 1])
            elif op.type == MaceOp.Reshape.name:
                for arg in op.arg:
                    if arg.name == MaceKeyword.mace_dim_str and \
                            len(arg.ints) == 4 and \
                            src_data_format == DataFormat.NCHW and \
                            has_data_format:
                        self.transpose_shape(arg.ints, [0, 2, 3, 1])
            elif op.type == MaceOp.Transpose.name:
                for arg in op.arg:
                    if arg.name == MaceKeyword.mace_dims_str and \
                            len(arg.ints) == 4 and \
                            src_data_format == DataFormat.NCHW and \
                            has_data_format:
                        dst_shape = [0, 3, 1, 2]
                        self.transpose_shape(dst_shape, arg.ints)
                        self.transpose_shape(dst_shape, [0, 2, 3, 1])
                        arg.ints[:] = dst_shape

            # transpose op output shape
            if src_data_format == DataFormat.NCHW and \
                    has_data_format:
                print("Transpose output shapes: %s(%s)" % (op.name, op.type))
                for output_shape in op.output_shape:
                    if len(output_shape.dims) == 4:
                        self.transpose_shape(output_shape.dims,
                                             [0, 2, 3, 1])

        return False

    def quantize_nodes(self):
        if not self._option.quantize:
            return False

        print("Add mace quantize and dequantize nodes")

        for op in self._model.op:
            for i in range(len(op.input)):
                if op.input[i] in self._option.input_nodes:
                    input_node = self._option.input_nodes[op.input[i]]
                    if input_node.data_type == mace_pb2.DT_INT32:
                        continue
                if op.input[i] in self.input_name_map:
                    op.input[i] = self.input_name_map[op.input[i]]
            for i in range(len(op.output)):
                if op.output[i] in self.output_name_map:
                    op.name = MaceKeyword.mace_output_node_name \
                              + '_' + op.name
                    new_output_name = self.output_name_map[op.output[i]]
                    self._quantize_activation_info[new_output_name] = \
                        self._quantize_activation_info[op.output[i]]
                    if op.output[i] in self._consumers:
                        for consumer_op in self._consumers[op.output[i]]:
                            self.replace(consumer_op.input,
                                         op.output[i],
                                         new_output_name)
                    op.output[i] = new_output_name

            data_type_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_op_data_type_str)
            mace_check(data_type_arg, "Data type does not exist for %s(%s)"
                       % (op.name, op.type))
            if data_type_arg.i == mace_pb2.DT_FLOAT:
                if self._option.quantize_schema == \
                        MaceKeyword.mace_apu_16bit_per_tensor:
                    data_type_arg.i = mace_pb2.DT_INT16
                elif self._option.quantize_schema == \
                        MaceKeyword.mace_htp_u16a_s8w:
                    data_type_arg.i = mace_pb2.DT_UINT16
                elif self._option.quantize_schema == MaceKeyword.mace_int8:
                    data_type_arg.i = mace_pb2.DT_INT8
                else:
                    data_type_arg.i = mace_pb2.DT_UINT8
            elif data_type_arg.i == mace_pb2.DT_UINT8:
                mace_check(op.type == MaceOp.Quantize.name
                           or op.type == MaceOp.Dequantize.name,
                           "Only Quantization ops support uint8, "
                           "but got %s(%s)" % (op.name, op.type))
            elif data_type_arg.i == mace_pb2.DT_INT16 \
                and self._option.quantize_schema == \
                    MaceKeyword.mace_apu_16bit_per_tensor:
                mace_check(op.type == MaceOp.Quantize.name
                           or op.type == MaceOp.Dequantize.name,
                           "Only Quantization ops support int16, "
                           "but got %s(%s)" % (op.name, op.type))
            elif data_type_arg.i == mace_pb2.DT_UINT16 \
                and self._option.quantize_schema == \
                    MaceKeyword.mace_htp_u16a_s8w:
                mace_check(op.type == MaceOp.Quantize.name
                           or op.type == MaceOp.Dequantize.name,
                           "Only Quantization ops support int16, "
                           "but got %s(%s)" % (op.name, op.type))
            elif data_type_arg.i == mace_pb2.DT_INT8 \
                and self._option.quantize_schema == \
                    MaceKeyword.mace_int8:
                mace_check(op.type == MaceOp.Quantize.name
                           or op.type == MaceOp.Dequantize.name,
                           "Only Quantization ops support int8, "
                           "but got %s(%s)" % (op.name, op.type))
            else:
                mace_check(op.type == MaceOp.Quantize.name,
                           "Quantization only support float ops, "
                           "but get %s(%s, %s)"
                           % (op.name, op.type,
                              mace_pb2.DataType.Name(data_type_arg.i)))

        for i, input_node in enumerate(self._option.input_nodes.values()):
            if input_node.data_type == mace_pb2.DT_INT32:
                continue
            new_input_name = self.input_name_map[input_node.name]
            op_def = self._model.op.add()
            op_def.name = self.normalize_op_name(new_input_name)
            op_def.type = MaceOp.Quantize.name
            op_def.input.extend([input_node.name])
            op_def.output.extend([new_input_name])
            output_shape = op_def.output_shape.add()
            output_shape.dims.extend(input_node.shape)
            quantize_info = self._quantize_activation_info[new_input_name]
            self.copy_quantize_info(op_def, quantize_info)
            self._model.input_info[i].scale = quantize_info.scale
            self._model.input_info[i].zero_point = quantize_info.zero_point

            if self._option.quantize_schema == \
                    MaceKeyword.mace_apu_16bit_per_tensor:
                ConverterUtil.add_data_type_arg(op_def, mace_pb2.DT_INT16)
            elif self._option.quantize_schema == \
                    MaceKeyword.mace_htp_u16a_s8w:
                ConverterUtil.add_data_type_arg(op_def, mace_pb2.DT_UINT16)
            elif self._option.quantize_schema == MaceKeyword.mace_int8:
                ConverterUtil.add_data_type_arg(op_def, mace_pb2.DT_INT8)
            else:
                ConverterUtil.add_data_type_arg(op_def, mace_pb2.DT_UINT8)
            ConverterUtil.add_data_format_arg(op_def, input_node.data_format)
            # use actual ranges for model input quantize
            find_range_every_time_arg = op_def.arg.add()
            find_range_every_time_arg.name = \
                MaceKeyword.mace_find_range_every_time
            find_range_every_time_arg.i = 1

        output_nodes = self._option.check_nodes.values()
        for i, output_node in enumerate(output_nodes):
            op_def = self._model.op.add()
            op_def.name = self.normalize_op_name(output_node.name)
            op_def.type = MaceOp.Dequantize.name
            op_def.input.extend([self.output_name_map[output_node.name]])
            op_def.output.extend([output_node.name])
            output_shape = op_def.output_shape.add()
            producer_op = self._producer[output_node.name]
            output_shape.dims.extend(producer_op.output_shape[0].dims)
            op_def.output_type.extend([mace_pb2.DT_FLOAT])
            quantize_info = producer_op.quantize_info[0]
            self._model.output_info[i].scale = quantize_info.scale
            self._model.output_info[i].zero_point = quantize_info.zero_point

            if self._option.quantize_schema == \
                    MaceKeyword.mace_apu_16bit_per_tensor:
                ConverterUtil.add_data_type_arg(op_def, mace_pb2.DT_INT16)
            elif self._option.quantize_schema == \
                    MaceKeyword.mace_htp_u16a_s8w:
                ConverterUtil.add_data_type_arg(op_def, mace_pb2.DT_UINT16)
            elif self._option.quantize_schema == MaceKeyword.mace_int8:
                ConverterUtil.add_data_type_arg(op_def, mace_pb2.DT_INT8)
            else:
                ConverterUtil.add_data_type_arg(op_def, mace_pb2.DT_UINT8)
            ConverterUtil.add_data_format_arg(op_def, output_node.data_format)

        quantize_flag_arg = self._model.arg.add()
        quantize_flag_arg.name = MaceKeyword.mace_quantize_flag_arg_str
        quantize_flag_arg.i = 1

        return False

    def quantize_tensor(self, tensor):
        """Assume biasadd has been already folded with convolution and fc"""
        if tensor.data_type == mace_pb2.DT_FLOAT:
            ops = self._consumers.get(tensor.name, None)
            check_conv = False
            check_deconv = False
            if ops is not None and len(ops) == 1:
                if len(ops[0].input) >= 3:
                    check_conv =\
                        ops[0].type in [MaceOp.Conv2D.name,
                                        MaceOp.DepthwiseConv2d.name,
                                        MaceOp.FullyConnected.name,
                                        MaceOp.MatMul.name]\
                        and ops[0].input[2] == tensor.name
                # in tensorflow deconv's bias is the forth input
                if ops[0].type in [MaceOp.Deconv2D.name,
                                   MaceOp.DepthwiseDeconv2d]:
                    from_caffe = ConverterUtil.get_arg(
                        ops[0],
                        MaceKeyword.mace_framework_type_str).i ==\
                                 FrameworkType.CAFFE.value
                    if from_caffe and len(ops[0].input) >= 3:
                        check_deconv = ops[0].input[2] == tensor.name
                    else:
                        if len(ops[0].input) >= 4:
                            check_deconv = ops[0].input[3] == tensor.name
            if check_conv or check_deconv:
                conv_op = ops[0]
                scale_input = self._quantize_activation_info[
                    conv_op.input[0]].scale
                if conv_op.input[1] not in self._quantized_tensor:
                    self.quantize_tensor(self._consts[conv_op.input[1]])
                scale_filter = self._consts[conv_op.input[1]].scale
                scale = scale_input * scale_filter
                quantized_tensor = \
                    quantize_util.quantize_with_scale_and_zero(
                        tensor.float_data, scale, 0)
                if self._option.device == DeviceType.HEXAGON.value or \
                        self._option.device == DeviceType.HTA.value:
                    quantized_tensor.minval = scale * (-2**31)
                    quantized_tensor.maxval = scale * (2**31 - 1)
                tensor.data_type = mace_pb2.DT_INT32
            elif self._option.quantize_schema == \
                    MaceKeyword.mace_apu_16bit_per_tensor:
                quantized_tensor = \
                    quantize_util.quantize_int16(tensor.float_data)
                tensor.data_type = mace_pb2.DT_INT16
            elif self._option.quantize_schema == MaceKeyword.mace_int8:
                quantized_tensor = quantize_util.quantize_int8(
                    tensor.float_data)
                tensor.data_type = mace_pb2.DT_INT8
            else:
                non_zero = self._option.device == DeviceType.CPU.value
                has_qat = False
                if InfoKey.has_qat in self._converter_info:
                    if tensor.name in self._converter_info[InfoKey.has_qat]:
                        has_qat = True
                if has_qat and self._option.platform.name == "ONNX":
                    mace_check(
                        tensor.name in self._converter_info[InfoKey.qat_type],
                        "ONNX model tensor {} has QAT info,"
                        " but QAT type info is missing.".format(tensor.name))
                    tensor_qat_type = self._converter_info[
                        InfoKey.qat_type][tensor.name]
                    mace_check((tensor_qat_type == QatType.SYMMETRIC.value or
                               tensor_qat_type == QatType.ASYMMETRIC.value),
                               "QAT type can only be SYMMETRIC or ASYMMETRIC,"
                               " but {} is got.".format(tensor_qat_type))
                    symmetric = (tensor_qat_type == QatType.SYMMETRIC.value)
                    if symmetric:
                        maxval = 127.5 * tensor.scale
                        minval = -maxval
                    else:
                        minval, maxval = quantize_util.scale_zero_to_min_max(
                            tensor.scale,
                            tensor.zero_point)
                    func = quantize_util.quantize_with_min_and_max
                    quantized_tensor = func(tensor.float_data,
                                            self._option.device,
                                            non_zero,
                                            minval,
                                            maxval)
                else:
                    func = quantize_util.quantize
                    quantized_tensor = func(tensor.float_data,
                                            self._option.device,
                                            non_zero)
                tensor.data_type = mace_pb2.DT_UINT8

            del tensor.float_data[:]
            tensor.int32_data.extend(quantized_tensor.data)
            tensor.scale = quantized_tensor.scale
            tensor.zero_point = quantized_tensor.zero
            tensor.minval = quantized_tensor.minval
            tensor.maxval = quantized_tensor.maxval
            tensor.quantized = True
            self._quantized_tensor.update([tensor.name])

        return False

    def quantize_weights(self):
        print("Quantize weights")
        net = self._model
        for tensor in net.tensors:
            self.quantize_tensor(tensor)

        return False

    def quantize_large_tensor(self, tensor):
        if tensor.data_type == mace_pb2.DT_FLOAT:
            ops = self._consumers.get(tensor.name, None)
            if ops is not None and len(ops) == 1:
                if ops[0].type in [MaceOp.Conv2D.name,
                                   MaceOp.FullyConnected.name,
                                   MaceOp.MatMul.name]:
                    quantized_tensor = \
                        quantize_util.quantize(tensor.float_data,
                                               self._option.device,
                                               False)
                    tensor.data_type = mace_pb2.DT_UINT8

                    del tensor.float_data[:]
                    tensor.int32_data.extend(quantized_tensor.data)
                    tensor.scale = quantized_tensor.scale
                    tensor.zero_point = quantized_tensor.zero
                    tensor.minval = quantized_tensor.minval
                    tensor.maxval = quantized_tensor.maxval
                    tensor.quantized = True
                    self._quantized_tensor.update([tensor.name])

    def quantize_large_weights(self):
        print("Quantize large weights")
        net = self._model
        for tensor in net.tensors:
            self.quantize_large_tensor(tensor)

        return False

    def add_quantize_info(self, op, minval, maxval):
        quantize_schema = self._option.quantize_schema
        if quantize_schema == MaceKeyword.mace_apu_16bit_per_tensor:
            maxval = max(abs(minval), abs(maxval))
            minval = -maxval
            scale = maxval / 2**15
            zero = 0
        elif quantize_schema == MaceKeyword.mace_htp_u16a_s8w:
            scale, zero, minval, maxval = \
                quantize_util.adjust_range_uint16(minval, maxval,
                                                  self._option.device,
                                                  non_zero=False)
        elif quantize_schema == MaceKeyword.mace_int8:
            scale, zero, minval, maxval = quantize_util.adjust_range_int8(
                minval, maxval)
        else:
            scale, zero, minval, maxval = \
                quantize_util.adjust_range(minval, maxval, self._option.device,
                                           non_zero=False)
        quantize_info = op.quantize_info.add()
        quantize_info.minval = minval
        quantize_info.maxval = maxval
        quantize_info.scale = scale
        quantize_info.zero_point = zero

        return quantize_info

    def copy_quantize_info(self, op, info):
        quantize_info = op.quantize_info.add()
        quantize_info.minval = info.minval
        quantize_info.maxval = info.maxval
        quantize_info.scale = info.scale
        quantize_info.zero_point = info.zero_point

    def transform_fake_quantize(self):
        # Quantize info from fixpoint fine tune
        print("Transform fake quantize")

        net = self._model
        for op in net.op:
            if op.type == 'FakeQuantWithMinMaxVars' or \
                   op.type == 'FakeQuantWithMinMaxArgs':
                if self._option.quantize and op.input[0] not in self._consts:
                    producer_op = self._producer[op.input[0]]
                    minval = ConverterUtil.get_arg(op, 'min').f
                    maxval = ConverterUtil.get_arg(op, 'max').f
                    quantize_info = \
                        self.add_quantize_info(producer_op, minval, maxval)
                    self._quantize_activation_info[op.input[0]] = quantize_info
                    # for add -> fakequant pattern
                    self._quantize_activation_info[op.output[0]] = \
                        quantize_info

                    print(op.input[0], op.output[0])
                op.type = MaceOp.Identity.name

        return False

    def rearrange_batch_to_space(self):
        if not self._option.quantize:
            return False

        # Put b2s after biasadd and relu
        for conv_op in self._model.op:
            if conv_op.type in [MaceOp.Conv2D.name,
                                MaceOp.DepthwiseConv2d.name] \
                    and self.consumer_count(conv_op.output[0]) == 1:
                b2s_op = self._consumers[conv_op.output[0]][0]
                if b2s_op.type == MaceOp.BatchToSpaceND.name \
                        and self.consumer_count(b2s_op.output[0]) == 1:
                    biasadd_or_act_op = self._consumers[b2s_op.output[0]][0]
                    if biasadd_or_act_op.type == MaceOp.BiasAdd.name:
                        biasadd_op = biasadd_or_act_op
                        if self.consumer_count(biasadd_op.output[0]) == 1 \
                                and self._consumers[biasadd_op.output[0]][0].type == MaceOp.Activation.name:  # noqa
                            act_op = self._consumers[biasadd_op.output[0]][0]
                            biasadd_op.input[0] = conv_op.output[0]
                            b2s_op.input[0] = act_op.output[0]
                            act_op.output_shape[0].dims[:] = conv_op.output_shape[0].dims[:]
                            for op in self._consumers[act_op.output[0]]:
                                self.replace(op.input,
                                             act_op.output[0],
                                             b2s_op.output[0])
                        else:
                            biasadd_op.input[0] = conv_op.output[0]
                            b2s_op.input[0] = biasadd_op.output[0]
                            for op in self._consumers[biasadd_op.output[0]]:
                                self.replace(op.input,
                                             biasadd_op.output[0],
                                             b2s_op.output[0])

                        print("Rearrange batch to space: %s(%s)"
                              % (b2s_op.name, b2s_op.type))
                        return True
                    elif biasadd_or_act_op.type == MaceOp.Activation.name:
                        act_op = biasadd_or_act_op
                        act_op.input[0] = conv_op.output[0]
                        b2s_op.input[0] = act_op.output[0]
                        for op in self._consumers[act_op.output[0]]:
                            self.replace(op.input,
                                         act_op.output[0],
                                         b2s_op.output[0])

                        print("Rearrange batch to space: %s(%s)"
                              % (b2s_op.name, b2s_op.type))
                        return True

        return False

    def add_quantize_tensor_range(self):
        # Quantize info from range statistics
        range_file = self._option.quantize_range_file
        quantize_schema = self._option.quantize_schema
        if range_file:
            print("Add quantize tensor range")
            post_quantize_info = {}
            with open(range_file) as f:
                for line in f:
                    tensor_name, minmax = line.split("@@")[:2]
                    min_val, max_val = [float(i) for i in
                                        minmax.strip().split(",")]
                    if (quantize_schema ==
                            MaceKeyword.mace_apu_16bit_per_tensor):
                        max_val = max(abs(min_val), abs(max_val))
                        min_val = -max_val
                        scale = max_val / 2**15
                        zero = 0
                    elif quantize_schema == MaceKeyword.mace_int8:
                        scale, zero, min_val, max_val = \
                            quantize_util.adjust_range_int8(min_val, max_val)
                    elif quantize_schema == MaceKeyword.mace_htp_u16a_s8w:
                        device = self._option.device
                        scale, zero, min_val, max_val = \
                            quantize_util.adjust_range_uint16(min_val,
                                                              max_val,
                                                              device,
                                                              non_zero=False)
                    else:
                        scale, zero, min_val, max_val = \
                            quantize_util.adjust_range(min_val, max_val,
                                                       self._option.device,
                                                       non_zero=False)
                    activation_info = mace_pb2.QuantizeActivationInfo()
                    activation_info.minval = min_val
                    activation_info.maxval = max_val
                    activation_info.scale = scale
                    activation_info.zero_point = zero
                    if tensor_name not in self._quantize_activation_info:
                        post_quantize_info[tensor_name] = activation_info

            for op in self._model.op:
                if op.name.find(MaceKeyword.mace_output_node_name) >= 0:
                    continue
                for output in op.output:
                    # Prefer quantize info from quantization-aware training
                    if output not in self._quantize_activation_info:
                        mace_check(output in post_quantize_info,
                                   "%s does not have quantize activation info"
                                   % op)
                        op.quantize_info.extend([post_quantize_info[output]])
                        self._quantize_activation_info[output] = \
                            post_quantize_info[output]

        if not self._option.quantize:
            return False

        print("Add default quantize info for input")
        for i, input_node in enumerate(self._option.input_nodes.values()):
            if input_node.data_type == mace_pb2.DT_INT32:
                continue
            new_input_name = self.input_name_map[input_node.name]
            if input_node.name not in self._quantize_activation_info:
                print("Input range %s: %s" % (input_node.name,
                                              str(input_node.range)))
                if quantize_schema == MaceKeyword.mace_apu_16bit_per_tensor:
                    maxval = max(abs(input_node.range[0]),
                                 abs(input_node.range[1]))
                    minval = -maxval
                    scale = maxval / 2**15
                    zero = 0
                elif quantize_schema == MaceKeyword.mace_htp_u16a_s8w:
                    scale, zero, minval, maxval = \
                        quantize_util.adjust_range_uint16(input_node.range[0],
                                                          input_node.range[1],
                                                          self._option.device,
                                                          non_zero=False)
                elif quantize_schema == MaceKeyword.mace_int8:
                    scale, zero, minval, maxval = \
                        quantize_util.adjust_range_int8(
                            input_node.range[0], input_node.range[1])
                else:
                    scale, zero, minval, maxval = \
                        quantize_util.adjust_range(input_node.range[0],
                                                   input_node.range[1],
                                                   self._option.device,
                                                   non_zero=False)
                quantize_info = \
                    mace_pb2.QuantizeActivationInfo()
                quantize_info.minval = minval
                quantize_info.maxval = maxval
                quantize_info.scale = scale
                quantize_info.zero_point = zero
                self._quantize_activation_info[new_input_name] = quantize_info
                input_op = self._producer[input_node.name]
                input_op.quantize_info.extend([quantize_info])
            else:
                self._quantize_activation_info[new_input_name] = \
                    self._quantize_activation_info[input_node.name]

        print("Add default quantize info for ops like Pooling, Softmax")
        for op in self._model.op:
            if op.type in [MaceOp.ExpandDims.name,
                           MaceOp.Pad.name,
                           MaceOp.Pooling.name,
                           MaceOp.Reduce.name,
                           MaceOp.Reshape.name,
                           MaceOp.ResizeBilinear.name,
                           MaceOp.Squeeze.name,
                           MaceOp.StridedSlice.name,
                           MaceOp.BatchToSpaceND.name,
                           MaceOp.SpaceToBatchND.name,
                           MaceOp.SpaceToDepth.name,
                           MaceOp.DepthToSpace.name,
                           MaceOp.Transpose.name]:
                del op.quantize_info[:]
                producer_op = self._producer[op.input[0]]
                if producer_op.output[0] in self._option.input_nodes:
                    new_input_name = self.input_name_map[producer_op.output[0]]
                    self.copy_quantize_info(
                        op, self._quantize_activation_info[new_input_name])
                else:
                    self.copy_quantize_info(op,
                                            producer_op.quantize_info[0])
                self._quantize_activation_info[op.output[0]] = \
                    op.quantize_info[0]
            elif (op.type == MaceOp.Concat.name
                  and (not op.quantize_info
                       or self._option.change_concat_ranges)):
                if op.quantize_info:
                    maxval = op.quantize_info[0].maxval
                    minval = op.quantize_info[0].minval
                    del op.quantize_info[:]
                else:
                    maxval = float("-inf")
                    minval = float("inf")
                for i in range(len(op.input)):
                    minval = min(minval, self._producer[op.input[i]].quantize_info[0].minval)  # noqa
                    maxval = max(maxval, self._producer[op.input[i]].quantize_info[0].maxval)  # noqa
                quantize_info = \
                    self.add_quantize_info(op, minval, maxval)
                self._quantize_activation_info[op.output[0]] = quantize_info
                if self._option.change_concat_ranges:
                    for i in range(len(op.input)):
                        producer_op = self._producer[op.input[i]]
                        del producer_op.quantize_info[:]
                        self.copy_quantize_info(producer_op, quantize_info)
                        self._quantize_activation_info[producer_op.output[0]] \
                            = producer_op.quantize_info[0]
            elif op.type == MaceOp.Activation.name:
                act_type = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_activation_type_str).s.decode()
                if act_type not in [ActivationType.TANH.name,
                                    ActivationType.SIGMOID.name,
                                    ActivationType.RELUX.name]:
                    continue
                del op.quantize_info[:]
                if act_type == ActivationType.TANH.name:
                    quantize_info = self.add_quantize_info(op, -1.0, 1.0)
                elif act_type == ActivationType.SIGMOID.name:
                    quantize_info = self.add_quantize_info(op, 0.0, 1.0)
                elif act_type == ActivationType.RELUX.name:
                    for arg in op.arg:
                        if arg.name == MaceKeyword.mace_activation_max_limit_str:
                            maxval = arg.f
                            minval = 0.0
                            quantize_info = self.add_quantize_info(op, minval, maxval)
                self._quantize_activation_info[op.output[0]] = quantize_info
            elif op.type == MaceOp.Softmax.name:
                del op.quantize_info[:]
                if self._option.device == DeviceType.APU.value:
                    mace_check(quantize_schema != MaceKeyword.mace_htp_u16a_s8w,
                               "mace_htp_u16a_s8w is not a valid quantize_schema for APU")
                    if quantize_schema == MaceKeyword.mace_apu_16bit_per_tensor:
                        quantize_info = self.add_quantize_info(op, 0.0, 1.0)
                    else:
                        # to comply with softmax scale constraints for APU
                        quantize_info = self.add_quantize_info(op, 0.0, 255.0/256.0)
                else:
                    quantize_info = self.add_quantize_info(op, 0.0, 1.0)
                self._quantize_activation_info[op.output[0]] = quantize_info
            elif (op.type == MaceOp.Eltwise.name
                  and not op.quantize_info
                  and len(op.input) == 2
                  and op.input[0] not in self._consts
                  and op.input[1] not in self._consts):
                producer_op0 = self._producer[op.input[0]]
                producer_op1 = self._producer[op.input[1]]
                if ConverterUtil.get_arg(
                        op, MaceKeyword.mace_element_type_str).i \
                        == EltwiseType.SUM.value:
                    minval = producer_op0.quantize_info[0].minval \
                        + producer_op1.quantize_info[0].minval
                    maxval = producer_op0.quantize_info[0].maxval \
                        + producer_op1.quantize_info[0].maxval
                elif ConverterUtil.get_arg(
                        op, MaceKeyword.mace_element_type_str).i \
                        == EltwiseType.SUB.value:
                    minval = producer_op0.quantize_info[0].minval \
                        - producer_op1.quantize_info[0].maxval
                    maxval = producer_op0.quantize_info[0].maxval \
                        - producer_op1.quantize_info[0].minval
                elif ConverterUtil.get_arg(
                        op, MaceKeyword.mace_element_type_str).i \
                        == EltwiseType.PROD.value:
                    mul_a = producer_op0.quantize_info[0].minval \
                        * producer_op1.quantize_info[0].minval
                    mul_b = producer_op0.quantize_info[0].minval \
                        * producer_op1.quantize_info[0].maxval
                    mul_c = producer_op0.quantize_info[0].maxval \
                        * producer_op1.quantize_info[0].minval
                    mul_d = producer_op0.quantize_info[0].maxval \
                        * producer_op1.quantize_info[0].maxval
                    minval = min(mul_a, mul_b, mul_c, mul_d)
                    maxval = max(mul_a, mul_b, mul_c, mul_d)
                else:
                    print(op)
                    mace_check(False, "Quantized Elementwise only support:"
                                      " SUM and SUB without ranges now.")
                quantize_info = \
                    self.add_quantize_info(op, minval, maxval)
                self._quantize_activation_info[op.output[0]] = quantize_info
            elif op.type == MaceOp.Split.name:
                del op.quantize_info[:]
                producer_op = self._producer[op.input[0]]
                for i in op.output:
                    self.copy_quantize_info(op,
                                            producer_op.quantize_info[0])
        return False

    def check_quantize_info(self):
        if not self._option.quantize:
            return False

        print("Check quantize info")
        for op in self._model.op:
            if (op.name.find(MaceKeyword.mace_input_node_name) == -1
                and op.name.find(MaceKeyword.mace_output_node_name) == -1
                and op.type != MaceOp.Quantize.name
                and op.type != MaceOp.Dequantize.name):  # noqa
                mace_check(len(op.output) == len(op.quantize_info),
                           "missing quantize info: %s" % op)
            for i in six.moves.range(len(op.quantize_info)):
                print("Op output %s range: [%f, %f]" % (
                    op.output[i],
                    op.quantize_info[i].minval,
                    op.quantize_info[i].maxval))

    def fp16_gather_weight(self):
        for op in self._model.op:
            if op.type != MaceOp.Gather.name:
                continue
            if op.input[0] not in self._consts:
                raise KeyError("Not in const tensor: " + str(op.input[0]))

            const_tensor = self._consts[op.input[0]]
            if const_tensor.data_type == mace_pb2.DT_FLOAT16:
                print(str(const_tensor.name) + " is alreay float16")
                continue

            print("FP16 Embedding Lookup Weights: %s" % const_tensor.name)

            op_outputs = [x for x in op.output]
            new_gather_name = op.name + '_fp16'
            new_gather_output_name = new_gather_name + ":0"
            dehalve_name = op.name

            # fp16 weights
            const_tensor.data_type = mace_pb2.DT_FLOAT16

            # change gather
            op.name = new_gather_name
            op.output[:] = [new_gather_output_name]
            # op.output.extend([new_gather_output_name])
            data_type_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_op_data_type_str)  # noqa
            if data_type_arg is None:
                data_type_arg = op.arg.add()
                data_type_arg.name = MaceKeyword.mace_op_data_type_str
            data_type_arg.i = mace_pb2.DT_FLOAT16

            # add dehalve
            dehalve_op = self._model.op.add()
            dehalve_op.name = dehalve_name
            dehalve_op.type = MaceOp.Cast.name
            dehalve_op.input.extend([new_gather_output_name])
            dehalve_op.output.extend(op_outputs)
            dehalve_op.output_shape.extend(op.output_shape)
            dehalve_op.output_type.extend([mace_pb2.DT_FLOAT])
            data_type_arg = dehalve_op.arg.add()
            data_type_arg.name = MaceKeyword.mace_op_data_type_str
            data_type_arg.i = mace_pb2.DT_FLOAT16

    def fp16_matmul_weight(self):
        if self._option.device != DeviceType.CPU.value:
            return

        print('Convert matmul weights to fp16 for specific matmul: activation + weights')  # noqa

        for op in self._model.op:
            if op.type != MaceOp.MatMul.name:
                continue
            if op.input[0] not in self._consts and op.input[1] not in self._consts:  # noqa
                continue
            if op.input[0] in self._consts and op.input[1] in self._consts:
                continue

            # Matmul fp16 Op only support fp32[1,k] x fp16[w,k]T or fp16[w,k] x fp32[k,1] now!  # noqa

            transpose_a_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_transpose_a_str)  # noqa
            transpose_b_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_transpose_b_str)  # noqa
            transpose_a = transpose_a_arg is not None and transpose_a_arg.i == 1  # noqa
            transpose_b = transpose_b_arg is not None and transpose_b_arg.i == 1  # noqa

            left_tensor = op.input[0]
            right_tensor = op.input[1]
            left_shape = self.get_tensor_shape(left_tensor)
            right_shape = self.get_tensor_shape(right_tensor)

            height = left_shape[-1] if transpose_a else left_shape[-2]
            width = right_shape[-2] if transpose_b else right_shape[-1]
            batch = reduce(lambda x, y: x * y, left_shape[: -2], 1)

            if batch != 1:
                continue

            if left_tensor in self._consts:
                if width != 1 or transpose_a:
                    continue
                const_tensor = self._consts[left_tensor]
            else:
                if height != 1 or not transpose_b:
                    continue
                const_tensor = self._consts[right_tensor]

            print('Convert Matmul Weights to fp16: %s' % op.name)

            const_tensor.data_type = mace_pb2.DT_FLOAT16
            data_type_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_op_data_type_str)  # noqa
            if data_type_arg is None:
                data_type_arg = op.arg.add()
                data_type_arg.name = MaceKeyword.mace_op_data_type_str
            data_type_arg.i = mace_pb2.DT_FLOAT16
            op.output_type.extend([mace_pb2.DT_FLOAT])

    def add_opencl_informations(self):
        print("Add OpenCL informations")
        net = self._model
        arg = net.arg.add()
        arg.name = MaceKeyword.mace_opencl_mem_type
        if self._option.cl_mem_type == "image":
            arg.i = MemoryType.GPU_IMAGE.value
        else:
            arg.i = MemoryType.GPU_BUFFER.value

    def transform_reshape_and_flatten(self):
        net = self._model
        for op in net.op:
            if op.type != MaceOp.Reshape.name:
                continue
            dim_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_dim_str)
            shape_tensor = None
            if len(op.input) == 1:
                print("Transform Caffe or PyTorch Reshape")
                dims = []
                axis_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str)
                # transform caffe reshape op
                if dim_arg:
                    dims = dim_arg.ints
                    shape_tensor = net.tensors.add()
                    shape_tensor.name = op.name + '_shape'
                    shape_tensor.dims.append(len(op.output_shape[0].dims))
                    shape_tensor.data_type = mace_pb2.DT_INT32
                # transform caffe flatten op
                elif axis_arg is not None:
                    axis = axis_arg.i
                    for i in range(0, axis):
                        dims.append(0)
                    dims.append(-1)
                    for i in range(axis + 1, len(op.output_shape[0].dims)):
                        dims.append(0)
                    shape_tensor = net.tensors.add()
                    shape_tensor.name = op.name + '_shape'
                    shape_tensor.dims.append(len(dims))
                    shape_tensor.data_type = mace_pb2.DT_INT32
                else:
                    mace_check(False, "Only support reshape and flatten")
                shape_tensor.int32_data.extend(dims)
                op.input.append(shape_tensor.name)

    def transform_shape_tensor_to_param(self):
        kOpTypeMap = {
            MaceOp.ResizeNearestNeighbor.name:
                (1, MaceKeyword.mace_resize_size_str),
            MaceOp.Deconv2D.name: (2, MaceKeyword.mace_dim_str),
            MaceOp.Reshape.name: (1, MaceKeyword.mace_dim_str),
        }
        net = self._model
        for op in net.op:
            if op.type not in kOpTypeMap:
                continue
            info = kOpTypeMap[op.type]
            dim_arg = ConverterUtil.get_arg(op, info[1])
            if len(op.input) > info[0] and dim_arg is None and \
                    op.input[info[0]] in self._consts:
                shape_tensor = self._consts[op.input[info[0]]]
                dim_arg = op.arg.add()
                dim_arg.name = info[1]
                dim_arg.ints.extend(shape_tensor.int32_data)

    def fold_fc_reshape(self):
        if self._option.device in [DeviceType.APU.value, DeviceType.HTP.value]:
            return False
        net = self._model
        for op in net.op:
            # whether to reshape fc output(default 4D)
            if op.type == MaceOp.FullyConnected.name and\
                    op.output[0] in self._consumers:
                consumers = self._consumers[op.output[0]]
                op_output_shape = op.output_shape[0].dims[:]
                for consumer in consumers:
                    if consumer.type == MaceOp.Reshape.name and \
                            consumer.input[1] in self._consts and \
                            self._consts[consumer.input[1]].int32_data[:] == \
                            [op_output_shape[0], 1, 1, op_output_shape[1]]:
                        # work for tensorflow
                        net.tensors.remove(self._consts[consumer.input[1]])
                        del consumer.input[1]
                        self.safe_remove_node(consumer, None)
                        return True
        return False

    def transform_channel_shuffle(self):
        net = self._model
        for op in net.op:
            if op.type == MaceOp.Transpose.name and \
                    len(op.output_shape[0].dims) == 5:
                perm = ConverterUtil.get_arg(op,
                                             MaceKeyword.mace_dims_str).ints
                framework = ConverterUtil.framework_type(net)
                if framework == FrameworkType.TENSORFLOW.value and \
                        [0, 1, 2, 4, 3] == list(perm):
                    group_dim = 4
                elif framework == FrameworkType.ONNX.value and \
                        [0, 2, 1, 3, 4] == list(perm):
                    group_dim = 2
                else:
                    continue

                # Remove the following Reshape op
                reshape_op = self._consumers.get(op.output[0], None)
                if (reshape_op and
                        len(reshape_op) == 1 and
                        reshape_op[0].type == MaceOp.Reshape.name and
                        len(reshape_op[0].output_shape[0].dims) == 4):
                    print("Transform channel shuffle")
                    output_shape = reshape_op[0].output_shape[0].dims
                    self.safe_remove_node(reshape_op[0], op,
                                          remove_input_tensor=True)
                else:
                    continue

                # Change Transpose op to ChannelShuffle
                op.type = MaceOp.ChannelShuffle.name
                del op.arg[:]
                group_arg = op.arg.add()
                group_arg.name = MaceKeyword.mace_group_str
                group_arg.i = op.output_shape[0].dims[group_dim]
                op.output_shape[0].dims[:] = output_shape

                # Remove previous Reshape op
                producer_op = self._producer.get(op.input[0], None)
                if producer_op:
                    if producer_op.type == MaceOp.Reshape.name:
                        self.safe_remove_node(producer_op, None)
                    elif producer_op.type == MaceOp.Stack.name:
                        print("Change channel shuffle stack to concat")
                        # Change previous Stack op to Concat if any
                        producer_op.type = MaceOp.Concat.name
                        producer_op.output_shape[0].dims[:] = output_shape

                return True

    def quantize_specific_ops_only(self):
        """
        This transform rule is only used internally, we are not gonna make
        things too complex for users
        """
        to_quantize_ops_output_type = {
            MaceOp.MatMul.name: mace_pb2.DT_INT32,
            MaceOp.Gather.name: mace_pb2.DT_UINT8,
        }

        for op in self._model.op:
            if (op.type not in to_quantize_ops_output_type
                    or len(op.output) > 1
                    or ConverterUtil.get_arg(op,
                                             MaceKeyword.mace_op_data_type_str).i != mace_pb2.DT_FLOAT):  # noqa
                # only support single output
                continue

            quantized_inputs_names = []

            should_quantize = False
            has_const = False
            for idx, input_tensor in enumerate(op.input):
                if input_tensor in self._consts:
                    has_const = True
                    break
            if not has_const:
                continue

            for idx, input_tensor in enumerate(op.input):
                if self.get_tensor_data_type(input_tensor) \
                        == mace_pb2.DT_FLOAT:
                    should_quantize = True
                    break
            if not should_quantize:
                continue
            else:
                print("Quantize op %s (%s)" % (op.name, op.type))

            non_zero = self._option.device == DeviceType.CPU.value \
                and op.type == MaceOp.MatMul.name

            for idx, input_tensor in enumerate(op.input):
                quantized_inputs_names.append(input_tensor)

                if self.get_tensor_data_type(input_tensor) \
                        != mace_pb2.DT_FLOAT:
                    continue

                if input_tensor in self._consts:
                    const_tensor = self._consts[input_tensor]
                    quantized_tensor = quantize_util.quantize(
                        const_tensor.float_data, self._option.device, non_zero)
                    del const_tensor.float_data[:]
                    const_tensor.int32_data.extend(quantized_tensor.data)
                    const_tensor.data_type = mace_pb2.DT_UINT8
                    const_tensor.scale = quantized_tensor.scale
                    const_tensor.zero_point = quantized_tensor.zero
                    const_tensor.minval = quantized_tensor.minval
                    const_tensor.maxval = quantized_tensor.maxval
                    const_tensor.quantized = True
                else:
                    input_shape = self.get_tensor_shape(input_tensor)
                    quantize_op = self._model.op.add()
                    quantize_op.name = self.normalize_op_name(
                        input_tensor) + "_quant"
                    quantize_op.type = MaceOp.Quantize.name
                    quantize_op.input.extend([input_tensor])
                    quantize_output_name = quantize_op.name + '_0'
                    quantize_op.output.extend([quantize_output_name])
                    output_shape = quantize_op.output_shape.add()
                    output_shape.dims.extend(input_shape)
                    quantize_op.output_type.extend([mace_pb2.DT_UINT8])
                    data_type_arg = quantize_op.arg.add()
                    data_type_arg.name = MaceKeyword.mace_op_data_type_str
                    data_type_arg.i = mace_pb2.DT_UINT8
                    ConverterUtil.add_data_format_arg(
                        quantize_op,
                        self.get_tensor_data_format(input_tensor))

                    data_type_arg = quantize_op.arg.add()
                    data_type_arg.name = MaceKeyword.mace_non_zero
                    data_type_arg.i = 0

                    find_range_arg = quantize_op.arg.add()
                    find_range_arg.name = \
                        MaceKeyword.mace_find_range_every_time
                    find_range_arg.i = 1

                    quantized_inputs_names[-1] = quantize_output_name

            del op.input[:]
            op.input.extend(quantized_inputs_names)

            original_output_name = op.output[0]
            op.output[0] = original_output_name + "_quant"
            op.output_type.extend([to_quantize_ops_output_type[op.type]])
            data_type_arg = ConverterUtil.get_arg(op,
                                                  MaceKeyword.mace_op_data_type_str)  # noqa
            if data_type_arg is None:
                data_type_arg = op.arg.add()
                data_type_arg.name = MaceKeyword.mace_op_data_type_str
            data_type_arg.i = mace_pb2.DT_UINT8

            dequantize_op = self._model.op.add()
            dequantize_op.name = op.name + "_dequant"
            dequantize_op.type = MaceOp.Dequantize.name
            dequantize_op.input.extend([op.output[0]])
            dequantize_op.output.extend([original_output_name])
            dequantize_op.output_shape.extend(op.output_shape)
            dequantize_op.output_type.extend([mace_pb2.DT_FLOAT])
            data_type_arg = dequantize_op.arg.add()
            data_type_arg.name = MaceKeyword.mace_op_data_type_str
            data_type_arg.i = to_quantize_ops_output_type[op.type]
            ConverterUtil.add_data_format_arg(
                dequantize_op,
                self.get_tensor_data_format(original_output_name))
            quantize_flag_arg = ConverterUtil.get_arg(self._model,
                                                      MaceKeyword.mace_quantize_flag_arg_str)  # noqa
            if quantize_flag_arg is None:
                quantize_flag_arg = self._model.arg.add()
                quantize_flag_arg.name = MaceKeyword.mace_quantize_flag_arg_str
                quantize_flag_arg.i = 1

            return True

        return False

    def transform_single_bn_to_depthwise_conv(self):
        for op in self._model.op:
            if op.type != MaceOp.BatchNorm.name:
                continue

            if len(op.input) != 3:
                continue

            producer = self._producer[op.input[0]]
            if producer.type in [MaceOp.Conv2D.name,
                                 MaceOp.Deconv2D.name,
                                 MaceOp.DepthwiseDeconv2d.name,
                                 MaceOp.DepthwiseConv2d.name,
                                 MaceOp.BatchToSpaceND.name]:
                continue

            op.type = MaceOp.DepthwiseConv2d.name
            padding_arg = op.arg.add()
            padding_arg.name = MaceKeyword.mace_padding_str
            padding_arg.i = PaddingMode.VALID.value
            strides_arg = op.arg.add()
            strides_arg.name = MaceKeyword.mace_strides_str
            strides_arg.ints.extend([1, 1])
            dilation_arg = op.arg.add()
            dilation_arg.name = MaceKeyword.mace_dilations_str
            dilation_arg.ints.extend([1, 1])
            for tensor in self._model.tensors:
                if tensor.name == op.input[1]:
                    tensor.dims[:] = [1, 1, 1, tensor.dims[0]]
                    break
            return True
        return False

    def transform_mul_max_to_prelu(self):
        if self._option.device != DeviceType.APU.value:
            return False
        net = self._model
        for op in net.op:
            if op.type != MaceOp.Eltwise.name or \
                    ConverterUtil.get_arg(
                        op, MaceKeyword.mace_element_type_str).i \
                    != EltwiseType.PROD.value or \
                    op.output[0] not in self._consumers:
                continue
            if len(op.input) != 1:
                continue
            consumer_op = self._consumers[op.output[0]][0]
            if consumer_op.type != MaceOp.Eltwise.name or \
                    ConverterUtil.get_arg(
                        consumer_op, MaceKeyword.mace_element_type_str).i \
                    != EltwiseType.MAX.value:
                continue
            if op.input[0] not in consumer_op.input:
                continue
            float_value_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_scalar_input_str)
            mace_check(float_value_arg is not None,
                       op.name + ': ' + MaceKeyword.mace_scalar_input_str +
                       ' value float should not be None')
            scalar = float_value_arg.f
            if scalar < 0:
                continue
            if scalar > 1:
                scalar = 1
            # Change Mul op to Prelu
            print("Change mul and max to prelu: %s(%s)" % (op.name, op.type))
            op.name = consumer_op.name
            op.output[0] = consumer_op.output[0]
            alpha_tensor = net.tensors.add()
            alpha_tensor.name = op.name + '_alpha'
            alpha_tensor.dims.append(1)
            alpha_tensor.data_type = mace_pb2.DT_FLOAT
            alpha_tensor.float_data.extend([scalar])
            op.input.extend([alpha_tensor.name])
            ConverterUtil.del_arg(op, MaceKeyword.mace_scalar_input_str)
            ConverterUtil.del_arg(
                op, MaceKeyword.mace_scalar_input_index_str)
            op.type = MaceOp.Activation.name
            type_arg = op.arg.add()
            type_arg.name = MaceKeyword.mace_activation_type_str
            type_arg.s = six.b(ActivationType.PRELU.name)
            self.replace_quantize_info(op, consumer_op)
            self.safe_remove_node(consumer_op, op)
            return True
        return False

    def transform_expand_dims_to_reshape(self):
        if self._option.device != DeviceType.APU.value:
            return False
        net = self._model
        for op in net.op:
            if op.type == MaceOp.ExpandDims.name:
                op.type = MaceOp.Reshape.name
                return True
        return False

    def quantize_fold_relu(self):
        if self._option.quantize_schema != MaceKeyword.mace_int8:
            return

        net = self._model

        for op in net.op:
            if op.type == MaceOp.Activation.name:
                act_type_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_activation_type_str)
                act_type = act_type_arg.s.decode()

                if act_type in ["RELU", "RELUX"]:
                    producer = self._producer[op.input[0]]
                    self.replace_quantize_info(producer, op)
                    self.safe_remove_node(op, producer)
                    return True

        return False

    def transform_keras_quantize_info(self):
        mace_check(self._option.platform == Platform.KERAS, "For KERAS models")
        changed = False
        for op in self._model.op:
            for i in range(len(op.quantize_info)):
                if not op.output[i] in self._quantize_activation_info:
                    self._quantize_activation_info[op.output[i]] = \
                        op.quantize_info[i]
                    changed = True

        return changed

    def fold_div_bn(self):
        net = self._model
        for op in net.op:
            if op.type == MaceOp.BatchNorm.name:
                scale = self._consts[op.input[1]]
                producer_op = self._producer[op.input[0]]
                if producer_op.type != MaceOp.Eltwise.name:
                    continue
                eltwise_type = ConverterUtil.get_arg(
                    producer_op, MaceKeyword.mace_element_type_str)
                if eltwise_type.i != EltwiseType.DIV.value:
                    continue
                if producer_op.input[1] not in self._consts:
                    continue
                divisor = self._consts[producer_op.input[1]]
                if divisor.data_type != mace_pb2.DT_FLOAT or \
                        scale.data_type != mace_pb2.DT_FLOAT:
                    continue
                scale_dims = scale.dims
                divisor_dims = divisor.dims
                df_op = ConverterUtil.data_format(op)
                df_producer = ConverterUtil.data_format(producer_op)
                df_nchw = DataFormat.NCHW
                dim_match = df_op == df_nchw and \
                    df_producer == df_nchw and len(scale_dims) == 1 and \
                    len(divisor_dims) == 4 and \
                    divisor_dims[1] == np.prod(np.array(divisor_dims)) and \
                    divisor_dims[1] == scale_dims[0]
                if not dim_match:
                    continue
                if not np.allclose(np.array(scale.float_data).reshape(-1),
                                   np.array(divisor.float_data).reshape(-1)):
                    continue
                if producer_op.input[0] not in self._producer:
                    continue
                producer_producer = self._producer[producer_op.input[0]]
                self.safe_remove_node(
                    producer_op, producer_producer, remove_input_tensor=True)
                del op.input[1]
                self._model.tensors.remove(scale)
                op.type = MaceOp.BiasAdd.name
        return False

    def add_general_info(self):
        # add runtime arg
        runtime_arg = self._model.arg.add()
        runtime_arg.name = MaceKeyword.mace_runtime_type_str
        runtime_arg.i = self._option.device
        # add net info
        self._model.name = self._option.name
        self._model.infer_order = self._option.order
        return False

    def tensor_is_used(self, tensor):
        for output in self._model.output_info:
            if tensor.name == output.name:
                return True

        for op in self._model.op:
            for input in op.input:
                if tensor.name == input:
                    return True

        return False

    def remove_unused_tensor(self):
        unused_tensors = []
        for tensor in self._model.tensors:
            if not self.tensor_is_used(tensor):
                unused_tensors.append(tensor)
        for ts in unused_tensors:
            self._model.tensors.remove(ts)

        return len(unused_tensors) != 0

    # A complex rule that folds broken `instance_norm` generated by TensorFlow.
    # Two cases are considered: 1. scale = True and zero = True:
    #      UpstreamOp
    #      | |  |
    #  +---+ |  v
    #  |     | Reduce(MEAN)
    #  |     | (op, whose type will be changed to InstanceNorm)
    #  |     |  |      |
    #  |     |  |      |
    #  |     v  v      +------------+
    #  |  SqrDiffMean               |
    #  | (sqr_diff_mean_op)         |
    #  |       |                    |
    #  |       v                    |
    #  | Eltwise(SUM)               |
    #  |(var_plus_epsilon_op)       |
    #  |       |                    |
    #  |       v                    |
    #  | Eltwise(POW) Tensor(scale) |
    #  | (rsqrt_op)     |           |
    #  |       |        |           |
    #  |       v        v           |
    #  |       Eltwise(PROD)--------+-+
    #  |    (rsqrt_mul_scale_op)    | |
    #  |       |                    | |
    #  |       |                    v v
    #  |       |            Eltwise(PROD)
    #  |       |          (second_mean_consumer_op)
    #  |       |                     |
    #  |       |   Tensor(offset)    |
    #  |       |              |      |
    #  v       v              v      v
    # Eltwise(PROD)          Eltwise(SUB)
    # (lhs_of_final_add) (rhs_of_final_add)
    #       |                 |
    #       |          +------+
    #       v          v
    #       Eltwise(SUM)
    #      (final_add_op)
    #           |
    #           v
    #     DownstreamOp
    # 2. scale = False and zero = False:
    #             UpstreamOp
    #             | |   |
    #   +---------+ |   v
    #   |           | Reduce(MEAN)
    #   |           | (op, whose type will be changed to InstanceNorm)
    #   |           |   |    |
    #   |           v   v    +----+
    #   |        SqrDiffMean      |
    #   |   (sqr_diff_mean_op)    |
    #   |             |           |
    #   |             v           |
    #   |       Eltwise(SUM)      |
    #   | (var_plus_epsilon_op)   |
    #   |             |           v
    #   |             v        Eltwise(NEG)
    #   |       Eltwise(POW)  (second_mean_consumer_op)
    #   |       (rsqrt_op)        |
    #   |        |      |         |
    #   |  +-----+      +---------+---+
    #   |  |                      |   |
    #   v  v                      v   v
    # Eltwise(PROD)             Eltwise(PROD)
    # (lhs_of_final_add)      (rhs_of_final_add)
    #      |                      |
    #      +-------+      +---------+
    #              |      |
    #              v      v
    #             Eltwise(SUM)
    #            (final_add_op)
    #                 |
    #                 v
    #             DownstreamOp
    # When possible, both cases are folded into:
    #  UpstreamOp
    #     |
    #     v
    #  InstanceNorm
    #     |
    #     v
    #  DownstreamOp
    def get_rhs_op_scale_true(self, rsqrt_op, second_mean_consumer_op):
        rhs_dict = dict()
        rhs_dict['is_in'] = False
        rsqrt_consumers = self._consumers.get(
            rsqrt_op.output[0], [])
        if len(rsqrt_consumers) != 1:
            return rhs_dict
        # pow(var + epsilon, -0.5) * scale
        rsqrt_mul_scale_op = rsqrt_consumers[0]
        if rsqrt_mul_scale_op.type != MaceOp.Eltwise.name:
            return rhs_dict
        elt_type = ConverterUtil.get_arg(
            rsqrt_mul_scale_op, MaceKeyword.mace_element_type_str).i
        scale_tensor_name = rsqrt_mul_scale_op.input[1]
        if not (len(rsqrt_mul_scale_op.input) == 2 and
                len(rsqrt_mul_scale_op.output) == 1 and
                scale_tensor_name in self._consts and
                elt_type == EltwiseType.PROD.value and
                rsqrt_mul_scale_op.output[0] in second_mean_consumer_op.input):  # noqa
            return rhs_dict
        right_mul_consumers = self._consumers.get(
            second_mean_consumer_op.output[0], [])
        if len(right_mul_consumers) != 1:
            return rhs_dict
        # offset - pow(var + epsilon, -0.5) * scale; rhs of final Add
        offset_minus_rsqrt_mean_op = right_mul_consumers[0]
        if offset_minus_rsqrt_mean_op.type != MaceOp.Eltwise.name:
            return rhs_dict
        elt_type = ConverterUtil.get_arg(
            offset_minus_rsqrt_mean_op,
            MaceKeyword.mace_element_type_str).i
        if not (len(offset_minus_rsqrt_mean_op.input) == 2 and
                len(offset_minus_rsqrt_mean_op.output) == 1 and
                offset_minus_rsqrt_mean_op.input[0] in self._consts and
                elt_type == EltwiseType.SUB.value):
            return rhs_dict
        offset_tensor_name = offset_minus_rsqrt_mean_op.input[0]
        # Just an alias, to have a same variable name as scale_offset = 0.
        rhs_of_final_add = offset_minus_rsqrt_mean_op
        rhs_dict['is_in'] = True
        rhs_dict['rsqrt_mul_scale_op'] = rsqrt_mul_scale_op
        rhs_dict['scale_tensor_name'] = scale_tensor_name
        rhs_dict['offset_tensor_name'] = offset_tensor_name
        rhs_dict['rhs_of_final_add'] = rhs_of_final_add
        return rhs_dict

    def get_rhs_op_scale_false(self, rsqrt_op, second_mean_consumer_op):
        rhs_dict = dict()
        rhs_dict['is_in'] = False
        neg_consumers = self._consumers.get(
            second_mean_consumer_op.output[0], [])
        if len(neg_consumers) != 1:
            return rhs_dict
        # -pow(var + epsilon, -0.5) * mean; rhs of final Add
        rsqrt_mul_neg_mean_op = neg_consumers[0]
        if rsqrt_mul_neg_mean_op.type != MaceOp.Eltwise.name:
            return rhs_dict
        elt_type = ConverterUtil.get_arg(
            rsqrt_mul_neg_mean_op, MaceKeyword.mace_element_type_str).i
        if not (len(rsqrt_mul_neg_mean_op.input) == 2 and
                len(rsqrt_mul_neg_mean_op.output) == 1 and
                elt_type == EltwiseType.PROD.value and
                rsqrt_op.output[0] in rsqrt_mul_neg_mean_op.input):
            return rhs_dict
        rhs_of_final_add = rsqrt_mul_neg_mean_op
        rhs_dict['is_in'] = True
        rhs_dict['rhs_of_final_add'] = rhs_of_final_add
        return rhs_dict

    def get_lhs_and_final_add(self, scale_offset, rhs_dict, op, rsqrt_op):
        lhs_dict = dict()
        lhs_dict['is_in'] = False
        rhs_of_final_add = rhs_dict['rhs_of_final_add']
        if scale_offset:
            rsqrt_mul_scale_op = rhs_dict['rsqrt_mul_scale_op']
            consumers_after_branch = self._consumers.get(
                rsqrt_mul_scale_op.output[0], [])
        else:
            consumers_after_branch = self._consumers.get(
                rsqrt_op.output[0], [])
        if len(consumers_after_branch) != 2:
            return lhs_dict
        lhs_of_final_add = None
        for consume_op in consumers_after_branch:
            if consume_op.input[0] == op.input[0]:
                # scale_zero =  True: x * pow(var + epsilon, -0.5) * scale
                # scale_zero = False: x * pow(var + epsilon, -0.5)
                lhs_of_final_add = consume_op
                break
        if lhs_of_final_add is None:
            return lhs_dict
        if not (lhs_of_final_add is not None and
                lhs_of_final_add.type == MaceOp.Eltwise.name):
            return lhs_dict
        elt_type = ConverterUtil.get_arg(
            lhs_of_final_add, MaceKeyword.mace_element_type_str).i
        lhs_consumers = self._consumers.get(lhs_of_final_add.output[0], [])
        if not (len(lhs_of_final_add.input) == 2 and
                len(lhs_of_final_add.output) == 1 and
                len(lhs_consumers) == 1 and
                rhs_of_final_add.output[0] in lhs_consumers[0].input and
                elt_type == EltwiseType.PROD.value):
            return lhs_dict
        # scale_zero =  True: x * pow(var + epsilon, -0.5) * scale + \
        #                     offset - pow(var + epsilon, -0.5) * mean
        # scale_zero = False: x * pow(var + epsilon, -0.5) + \
        #                     pow(var + epsilon, -0.5) * (-mean)
        final_add_op = lhs_consumers[0]
        if not (final_add_op.type == MaceOp.Eltwise.name and
                len(final_add_op.input) == 2 and
                len(final_add_op.output) == 1):
            return lhs_dict
        elt_type = ConverterUtil.get_arg(
            final_add_op, MaceKeyword.mace_element_type_str).i
        if elt_type != EltwiseType.SUM.value:
            return lhs_dict
        lhs_dict['is_in'] = True
        lhs_dict['lhs_of_final_add'] = lhs_of_final_add
        lhs_dict['final_add_op'] = final_add_op
        return lhs_dict

    def do_fold_instance_norm(self, op, lhs_dict, rhs_dict, scale_offset,
                              unused_ops, unused_args, epsilon):
        net = self._model
        del op.output_shape[0].dims[:]
        op.output_shape[0].dims.extend(
            lhs_dict['final_add_op'].output_shape[0].dims)
        for unused_op in unused_ops:
            net.op.remove(unused_op)

        affine_arg = op.arg.add()
        affine_arg.name = MaceKeyword.mace_affine_str
        if scale_offset:
            affine_arg.i = 1
            op.input.extend([rhs_dict['scale_tensor_name'],
                            rhs_dict['offset_tensor_name']])
            net.op.remove(rhs_dict['rsqrt_mul_scale_op'])
        else:
            affine_arg.i = 0
        net.op.remove(rhs_dict['rhs_of_final_add'])
        net.op.remove(lhs_dict['lhs_of_final_add'])
        self.replace_quantize_info(op,  lhs_dict['final_add_op'])
        op.output[0] = lhs_dict['final_add_op'].output[0]
        self.safe_remove_node(lhs_dict['final_add_op'], op)

        op.type = MaceOp.InstanceNorm.name
        for arg in unused_args:
            op.arg.remove(arg)
        epsilon_arg = op.arg.add()
        epsilon_arg.name = MaceKeyword.mace_epsilon_str
        epsilon_arg.f = epsilon

    def fold_instance_norm(self):
        net = self._model
        for op in net.op:
            is_reduce = (op.type == MaceOp.Reduce.name and
                         len(op.input) == 1 and
                         len(op.output) == 1)
            if not is_reduce:
                continue
            reduce_type_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_reduce_type_str)
            reduce_type = reduce_type_arg.i
            axis_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str)
            axis = axis_arg.ints
            keepdims_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_keepdims_str)
            keepdims = keepdims_arg.i
            if not (reduce_type == ReduceType.MEAN.value and
                    len(axis) == 2 and
                    axis[0] == 1 and axis[1] == 2 and
                    keepdims == 1 and
                    len(op.output_shape[0].dims) == 4):
                continue
            # Ops that take Mean as input
            mean_consumers = self._consumers.get(op.output[0], [])
            if len(mean_consumers) != 2:
                continue
            sqr_diff_mean_idx = -1
            sqr_diff_mean_op = None
            for idx in range(2):
                if mean_consumers[idx].type == MaceOp.SqrDiffMean.name:
                    sqr_diff_mean_idx = idx
                    sqr_diff_mean_op = mean_consumers[idx]
                    break
            if sqr_diff_mean_idx == -1:
                # SqrDiffMean is not found, it is not InstanceNorm
                continue
            second_mean_consumer_op = mean_consumers[1 - sqr_diff_mean_idx]
            if second_mean_consumer_op.type != MaceOp.Eltwise.name:
                continue
            elt_type = ConverterUtil.get_arg(
                second_mean_consumer_op, MaceKeyword.mace_element_type_str).i
            scale_offset = False
            # second consumer of Mean can only be NEG or PROD, otherwise,
            # it's not InstanceNorm
            if elt_type == EltwiseType.PROD.value:
                scale_offset = True
            elif elt_type == EltwiseType.NEG.value:
                scale_offset = False
            else:
                continue
            # var + epsilon
            sqr_diff_mean_consumers = self._consumers.get(
                sqr_diff_mean_op.output[0], [])
            if len(sqr_diff_mean_consumers) != 1:
                continue
            var_plus_epsilon_op = sqr_diff_mean_consumers[0]
            if var_plus_epsilon_op.type != MaceOp.Eltwise.name:
                continue
            elt_type = ConverterUtil.get_arg(
                var_plus_epsilon_op, MaceKeyword.mace_element_type_str).i
            scalar_input_index = ConverterUtil.get_arg(
                var_plus_epsilon_op, MaceKeyword.mace_scalar_input_index_str).i
            if not (len(var_plus_epsilon_op.input) == 1 and
                    len(var_plus_epsilon_op.output) == 1 and
                    elt_type == EltwiseType.SUM.value and
                    scalar_input_index == 1):
                continue
            epsilon = ConverterUtil.get_arg(
                var_plus_epsilon_op, MaceKeyword.mace_scalar_input_str).f
            # 1 / sqrt(var + epsilon) = pow(var + epsilon, -0.5)
            var_plus_epsilon_consumers = self._consumers.get(
                var_plus_epsilon_op.output[0], [])
            if len(var_plus_epsilon_consumers) != 1:
                continue
            rsqrt_op = var_plus_epsilon_consumers[0]
            if rsqrt_op.type != MaceOp.Eltwise.name:
                continue
            elt_type = ConverterUtil.get_arg(
                rsqrt_op, MaceKeyword.mace_element_type_str).i
            power = ConverterUtil.get_arg(
                rsqrt_op, MaceKeyword.mace_scalar_input_str).f
            if not (len(rsqrt_op.input) == 1 and
                    len(rsqrt_op.output) == 1 and
                    elt_type == EltwiseType.POW.value and
                    power == -0.5):
                continue

            if scale_offset:
                rhs_dict = self.get_rhs_op_scale_true(
                    rsqrt_op, second_mean_consumer_op)
            else:
                rhs_dict = self.get_rhs_op_scale_false(
                    rsqrt_op, second_mean_consumer_op)
            if not rhs_dict['is_in']:
                continue
            lhs_dict = self.get_lhs_and_final_add(
                scale_offset, rhs_dict, op, rsqrt_op)
            if not lhs_dict['is_in']:
                continue
            # we have excluded all cases that are not instance_norm
            unused_ops = [sqr_diff_mean_op, var_plus_epsilon_op, rsqrt_op,
                          second_mean_consumer_op]
            unused_args = [reduce_type_arg, axis_arg, keepdims_arg]
            self.do_fold_instance_norm(op, lhs_dict, rhs_dict, scale_offset,
                                       unused_ops, unused_args, epsilon)
            return True

        return False

    # Some frameworks use `NCHW` dataformat, transpose and store const Tensor
    # of 4D in `NHWC` dataformat in disk. Thus, we have uniform
    # const Tensor dataform in disk.
    # For GPU, `NHWC` just works;
    # for CPU, `NHWC` -> `NCHW` is done in init stage.
    def do_single_transpose(self, input_name, already_dealt):
        tensor = self._consts[input_name]
        shape = list(tensor.dims)
        if len(shape) == 4:
            array = np.array(tensor.float_data).reshape(shape)
            array = array.transpose(0, 2, 3, 1)
            tensor.dims[:] = array.shape
            tensor.float_data[:] = array.flat
        already_dealt.add(input_name)

    def transpose_const_op_input(self):
        net = self._model
        framework = ConverterUtil.framework_type(net)
        is_onnx = (framework == FrameworkType.ONNX.value)
        is_torch = (framework == FrameworkType.PYTORCH.value)
        is_megengine = (framework == FrameworkType.MEGENGINE.value)
        already_dealt = set()
        equal_types = set([MaceOp.Eltwise.name, MaceOp.Concat.name])
        for op in net.op:
            if (is_onnx or is_torch or is_megengine) and \
                    not self._option.quantize and \
                    (self._option.device == DeviceType.GPU.value or
                     self._option.device == DeviceType.CPU.value):
                num_input = 1
                if op.type in equal_types:
                    num_input = len(op.input)
                for idx in range(num_input):
                    input_name = op.input[idx]
                    if (input_name in self._consts) and \
                            (input_name not in self._option.input_nodes) and \
                            (input_name not in already_dealt):
                        self.do_single_transpose(input_name, already_dealt)
        return False

    def transform_biasadd_to_add(self):
        if self._option.device != DeviceType.HTP.value:
            return False
        net = self._model
        for op in net.op:
            if (op.type == MaceOp.BiasAdd.name
                    and len(op.input) == 2
                    and op.input[1] in self._consts
                    and len(self._consts[op.input[1]].dims) == 1):
                print("Transform biasadd to add: %s(%s)" % (op.name, op.type))
                op.type = MaceOp.Eltwise.name
                type_arg = op.arg.add()
                type_arg.name = MaceKeyword.mace_element_type_str
                type_arg.i = EltwiseType.SUM.value
                return True

        return False

    def transform_slice_to_strided_slice(self):
        if self._option.device != DeviceType.HTP.value:
            return False
        net = self._model
        framework = ConverterUtil.framework_type(net)
        for op in net.op:
            if (op.type == MaceOp.Slice.name
                    and framework == FrameworkType.ONNX.value
                    and len(op.input) == 5):
                op.type = MaceOp.StridedSlice.name
                tensor_shape = self.get_tensor_shape(op.input[0])
                input3 = self._consts[op.input[3]]
                axes_data = input3.int32_data
                for tensor in self._model.tensors:
                    if tensor.name in [op.input[1], op.input[2], op.input[4]]:
                        tensor.dims[:] = [len(tensor_shape)]
                for tensor in self._model.tensors:
                    if tensor.name == op.input[1]:
                        for i in range(len(tensor_shape)):
                            if i not in axes_data:
                                tensor.int32_data.insert(i, 0)
                    if tensor.name == op.input[2]:
                        for i in range(len(tensor_shape)):
                            if i in axes_data:
                                if tensor.int32_data[i] < 0:
                                    tensor.int32_data[i] += tensor_shape[i] + 1
                            else:
                                tensor.int32_data.insert(i, tensor_shape[i])
                    if tensor.name == op.input[4]:
                        for i in range(len(tensor_shape)):
                            if i not in axes_data:
                                tensor.int32_data.insert(i, 1)
                del input3.int32_data[0]
                for tensor in self._model.tensors:
                    if tensor.name == op.input[1]:
                        for i in range(len(tensor_shape)):
                            input3.int32_data.insert(i, tensor.int32_data[i])
                    if tensor.name == op.input[2]:
                        for i in range(len(tensor_shape)):
                            input3.int32_data.insert(2*i+1, tensor.int32_data[i])
                    if tensor.name == op.input[4]:
                        for i in range(len(tensor_shape)):
                            input3.int32_data.insert(3*i+2, tensor.int32_data[i])
                np.array(input3.int32_data).reshape([len(tensor_shape), 3])
                input3.dims[:] = [len(tensor_shape), 3]
                return True

        return False

    def add_transpose_op(self, node, transpose_dims):
        producer_op = self._producer[node]
        op = self._model.op.add()
        op.name = node + "_transpose"
        op.type = MaceOp.Transpose.name
        op.input.append(node)
        tensor_name = op.input[0] + "_transpose"
        op.output.append(tensor_name)

        output_shape = op.output_shape.add()
        shape_info = producer_op.output_shape[0].dims
        transposed_shape = []
        for i in transpose_dims:
            transposed_shape.append(shape_info[i])
        output_shape.dims.extend(transposed_shape)
        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        data_type_arg.i = self._option.data_type
        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.ONNX.value
        ConverterUtil.add_data_format_arg(op, DataFormat.NONE)
        dims_arg = op.arg.add()
        dims_arg.name = MaceKeyword.mace_dims_str
        dims_arg.ints.extend(transpose_dims)

    def add_transpose_for_htp(self):
        if self._option.device != DeviceType.HTP.value:
            return False
        net = self._model
        framework = ConverterUtil.framework_type(net)
        for op in net.op:
            data_format = ConverterUtil.get_arg(
                op, MaceKeyword.mace_data_format_str)
            if op.input[0] in self._producer:
                producer_op = self._producer[op.input[0]]
                producer_data_format = ConverterUtil.get_arg(
                    producer_op, MaceKeyword.mace_data_format_str)
                if (op.type == MaceOp.Conv2D.name
                        and framework == FrameworkType.ONNX.value
                        and data_format.i == DataFormat.AUTO.value
                        and producer_data_format.i == DataFormat.NCHW.value):
                    self.add_transpose_op(op.input[0], [0, 2, 3, 1])
                    op.input[0] = op.input[0] + "_transpose"
                    data_format = ConverterUtil.get_arg(
                        op, MaceKeyword.mace_data_format_str)
                    data_format.i = DataFormat.NHWC.value
                    return True
                elif (op.type == MaceOp.MatMul.name
                        and framework == FrameworkType.ONNX.value
                        and data_format.i == DataFormat.NCHW.value
                        and producer_data_format.i == DataFormat.AUTO.value):
                    self.add_transpose_op(op.input[0], [0, 3, 1, 2])
                    op.input[0] = op.input[0] + "_transpose"
                    data_format = ConverterUtil.get_arg(
                        op, MaceKeyword.mace_data_format_str)
                    data_format.i = DataFormat.NONE.value
                    return True
                elif (op.type == MaceOp.Transpose.name
                        and framework == FrameworkType.ONNX.value
                        and data_format.i == DataFormat.NCHW.value
                        and producer_data_format.i == DataFormat.AUTO.value):
                    if op.output[0] in self._consumers:
                        consumer = self._consumers[op.output[0]][0]
                        if consumer.type == MaceOp.Reshape:
                            self.add_transpose_op(op.input[0], [0, 3, 1, 2])
                            op.input[0] = op.input[0] + "_transpose"
                            data_format = ConverterUtil.get_arg(
                                op, MaceKeyword.mace_data_format_str)
                            data_format.i = DataFormat.NONE.value
                            return True
                elif (op.type in [MaceOp.Eltwise.name, MaceOp.Concat.name]
                        and framework == FrameworkType.ONNX.value
                        and data_format.i == DataFormat.NCHW.value
                        and len(op.input) == 2
                        and op.input[1] in self._producer):
                    input_1 = self._producer[op.input[1]]
                    input_1_data_format = ConverterUtil.get_arg(
                        input_1, MaceKeyword.mace_data_format_str)
                    if producer_data_format.i == DataFormat.AUTO.value:
                        self.add_transpose_op(op.input[0], [0, 3, 1, 2])
                        op.input[0] = op.input[0] + "_transpose"
                        data_format = ConverterUtil.get_arg(
                            op, MaceKeyword.mace_data_format_str)
                        data_format.i = DataFormat.NONE.value
                        return True
                    elif input_1_data_format.i == DataFormat.AUTO.value:
                        print(op.input[1])
                        self.add_transpose_op(op.input[1], [0, 3, 1, 2])
                        op.input[1] = op.input[1] + "_transpose"
                        data_format = ConverterUtil.get_arg(
                            op, MaceKeyword.mace_data_format_str)
                        data_format.i = DataFormat.NONE.value
                        return True
        return False
