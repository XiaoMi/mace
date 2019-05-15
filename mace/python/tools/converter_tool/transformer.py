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

from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.converter_tool.base_converter import DataFormat
from mace.python.tools.converter_tool.base_converter import DeviceType
from mace.python.tools.converter_tool.base_converter import EltwiseType
from mace.python.tools.converter_tool.base_converter import FrameworkType
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import MaceFixedDataFormatOps  # noqa
from mace.python.tools.converter_tool.base_converter import MaceTransposableDataFormatOps  # noqa
from mace.python.tools.converter_tool.base_converter import PaddingMode
from mace.python.tools.converter_tool.base_converter import ReduceType
from mace.python.tools.converter_tool.base_converter import TransformerRule
from mace.python.tools.convert_util import mace_check
from mace.python.tools.quantization import quantize_util


class Transformer(base_converter.ConverterInterface):
    """A class for transform naive mace model to optimized model.
    This Transformer should be platform irrelevant. So, do not assume
    tensor name has suffix like ':0".
    """

    def __init__(self, option, model):
        # Dependencies
        # (TRANSFORM_MATMUL_TO_FC, TRANSFORM_GLOBAL_CONV_TO_FC) -> RESHAPE_FC_WEIGHT  # noqa
        self._registered_transformers = {
            TransformerRule.TRANSFORM_FAKE_QUANTIZE:
                self.transform_fake_quantize,
            TransformerRule.REMOVE_IDENTITY_OP: self.remove_identity_op,
            TransformerRule.TRANSFORM_GLOBAL_POOLING:
                self.transform_global_pooling,
            TransformerRule.TRANSFORM_LSTMCELL_ZEROSTATE:
                self.transform_lstmcell_zerostate,
            TransformerRule.TRANSFORM_BASIC_LSTMCELL:
                self.transform_basic_lstmcell,
            TransformerRule.FOLD_RESHAPE: self.fold_reshape,
            TransformerRule.TRANSFORM_MATMUL_TO_FC:
                self.transform_matmul_to_fc,
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
            TransformerRule.REARRANGE_BATCH_TO_SPACE:
                self.rearrange_batch_to_space,
            TransformerRule.FLATTEN_ATROUS_CONV: self.flatten_atrous_conv,
            TransformerRule.FOLD_ACTIVATION: self.fold_activation,
            TransformerRule.FOLD_SQRDIFF_MEAN: self.fold_squared_diff_mean,
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
            TransformerRule.TRANSPOSE_DATA_FORMAT: self.transpose_data_format,
            TransformerRule.CHECK_QUANTIZE_INFO:
                self.check_quantize_info,
            TransformerRule.TRANSPOSE_CAFFE_RESHAPE_AND_FLATTEN:
                self.transform_caffe_reshape_and_flatten,
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
        self.initialize_name_map()

    def run(self):
        for key in self._option.transformer_option:
            transformer = self._registered_transformers[key]
            while True:
                self.construct_ops_and_consumers(key)
                changed = transformer()
                if not changed:
                    break
        self.delete_after_check_nodes()
        return self._model, self._quantize_activation_info

    def initialize_name_map(self):
        for input_node in self._option.input_nodes.values():
            new_input_name = MaceKeyword.mace_input_node_name \
                             + '_' + input_node.name
            self.input_name_map[input_node.name] = new_input_name

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
            mace_check(len(op.output) == 1 and len(op.input) == 1,
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
                for consumer_op in self._consumers.get(op.output[i], []):
                    self.replace(consumer_op.input,
                                 op.output[i],
                                 replace_op.output[i])

            # if the op is output node, change replace_op output name to the op
            # output name
            for i in six.moves.range(len(op.output)):
                if op.output[i] in self._option.output_nodes:
                    for consumer in self._consumers.get(
                            replace_op.output[i], []):
                        self.replace(consumer.input,
                                     replace_op.output[i],
                                     op.output[i])
                    replace_op.output[i] = op.output[i]

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
            input_info.data_format = input_node.data_format.value
            input_info.dims.extend(input_node.shape)
            input_info.data_type = input_node.data_type

        output_nodes = self._option.check_nodes.values()
        for output_node in output_nodes:
            output_info = net.output_info.add()
            output_info.name = output_node.name
            output_info.data_format = output_node.data_format.value
            output_info.dims.extend(
                self._producer[output_node.name].output_shape[0].dims)
            output_info.data_type = output_node.data_type

        return False

    def remove_identity_op(self):
        net = self._model
        for op in net.op:
            if op.type == 'Identity':
                print("Remove identity: %s(%s)" % (op.name, op.type))
                self.safe_remove_node(op,
                                      self._producer.get(op.input[0], None))
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
            if (op.type == MaceOp.Eltwise.name
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
                            self.replace_quantize_info(op, consumer_op)
                            self.safe_remove_node(consumer_op, op)
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
                if consumer_op.type == MaceOp.BatchNorm.name and \
                        (input_len == 2 or
                         (input_len == 3 and op.input[-1] in self._consts)):
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
                if consumer_op.type == MaceOp.BatchNorm.name and (
                        (framework == FrameworkType.CAFFE.value and
                         (input_len == 2 or (input_len == 3 and
                                             op.input[-1] in self._consts))) or
                        (framework == FrameworkType.TENSORFLOW.value and
                         (input_len == 3 or (input_len == 4 and
                                             op.input[-1] in self._consts)))):
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
                if consumer_op.type == MaceOp.BatchNorm.name and \
                        (input_len == 2 or
                         (input_len == 3 and op.input[-1] in self._consts)):
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
        net = self._model
        for op in net.op:
            if (op.type == 'Eltwise'
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
            if (((op.type == MaceOp.Conv2D.name
                  or op.type == MaceOp.DepthwiseConv2d.name
                  or op.type == MaceOp.FullyConnected.name)
                 and len(op.input) == 2)
                or (op.type == MaceOp.Deconv2D.name
                    and ((ConverterUtil.get_arg(
                                op,
                                MaceKeyword.mace_framework_type_str).i ==
                          FrameworkType.CAFFE.value
                          and len(op.input) == 2)
                         or (ConverterUtil.get_arg(
                                        op,
                                        MaceKeyword.mace_framework_type_str).i
                             == FrameworkType.TENSORFLOW.value
                             and len(op.input) == 3)))) \
                    and len(self._consumers.get(op.output[0], [])) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                if consumer_op.type == MaceOp.BiasAdd.name:
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
               and self._option.device != DeviceType.APU.value:
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
                        else:
                            padding_arg.i = PaddingMode.VALID.value

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

                        self.safe_remove_node(op, None)
                        self.safe_remove_node(b2s_op, conv_op)
                        return True
        return False

    def fold_activation(self):
        net = self._model
        for op in net.op:
            if (op.type == MaceOp.Conv2D.name
                or op.type == MaceOp.Deconv2D.name
                or op.type == MaceOp.DepthwiseConv2d.name
                or op.type == MaceOp.FullyConnected.name
                or op.type == MaceOp.BatchNorm.name) \
                    and len(self._consumers.get(op.output[0], [])) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                if consumer_op.type == MaceOp.Activation.name \
                        and ConverterUtil.get_arg(
                            consumer_op,
                            MaceKeyword.mace_activation_type_str).s != 'PRELU':
                    print("Fold activation: %s(%s)" % (op.name, op.type))
                    op.name = consumer_op.name
                    op.output[0] = consumer_op.output[0]
                    for arg in consumer_op.arg:
                        if arg.name == MaceKeyword.mace_activation_type_str \
                                or arg.name == \
                                    MaceKeyword.mace_activation_max_limit_str \
                                or arg.name == MaceKeyword.mace_activation_leakyrelu_coefficient_str:  # noqa
                            op.arg.extend([arg])

                    self.replace_quantize_info(op, consumer_op)
                    self.safe_remove_node(consumer_op, op)
                    return True

        return False

    def transform_global_conv_to_fc(self):
        """Transform global conv to fc should be placed after transposing
        input/output and filter"""

        if self._option.quantize:
            return

        net = self._model
        for op in net.op:
            if op.type == MaceOp.Conv2D.name \
                    and len(op.input) >= 2 \
                    and op.input[1] in self._consts:
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
                        and zero_padding:
                    print("transform global conv to fc %s(%s)"
                          % (op.name, op.type))
                    op.type = MaceOp.FullyConnected.name

        return False

    def reshape_fc_weight(self):
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
                        filter = self._consts[rhs]
                        filter_data = np.array(filter.float_data).reshape(
                            filter.dims)
                        filter_data = filter_data.transpose(1, 0)
                        filter.float_data[:] = filter_data.flat
                        filter.dims[:] = filter_data.shape
                        arg.i = 1
                        six.print_('Transpose matmul weight to shape:',
                                   filter.dims)

    def transpose_filters(self):
        net = self._model
        filter_format = self.filter_format()
        transposed_filter = set()
        transposed_deconv_filter = set()

        if self._option.quantize and \
                (self._option.device == DeviceType.CPU.value or
                 self._option.device == DeviceType.APU.value):
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
                     self._option.device == DeviceType.APU.value)) and\
                        op.input[1] not in transposed_filter:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(transpose_order)
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
        elif self._option.quantize and \
                (self._option.device == DeviceType.HEXAGON.value or
                 self._option.device == DeviceType.HTA.value):
            print("Transpose filters to HWIO/HWIM")
            mace_check(filter_format == DataFormat.HWIO,
                       "HEXAGON only support HWIO/HWIM filter format.")
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
            # transform `input(4D) -> reshape(2D) -> matmul` to `fc(2D)`
            # fc output is 2D in transformer, using as 4D in op kernel
            # work for TensorFlow
            if op.type == MaceOp.Reshape.name and \
                    len(op.input) == 2 and \
                    op.input[1] in self._consts and \
                    len(op.output_shape[0].dims) == 2 and \
                    filter_format == DataFormat.HWIO and \
                    op.input[0] in self._producer:
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
                            if len(weight.dims) != 2 or \
                               weight.dims[0] != op.output_shape[0].dims[1]:
                                is_fc = False
                    if is_fc:
                        print('convert reshape and matmul to fc')
                        self.safe_remove_node(op, input_op,
                                              remove_input_tensor=True)
                        for matmul_op in consumers:
                            weight = self._consts[matmul_op.input[1]]
                            matmul_op.type = MaceOp.FullyConnected.name
                            weight_data = np.array(weight.float_data).reshape(
                                weight.dims)
                            weight.dims[:] = input_shape[1:] + \
                                [weight_data.shape[1]]
                        return True

            # transform `fc1(2D) -> matmul` to `fc1(2D) -> fc1(2D)`
            if op.type == MaceOp.MatMul.name and \
                    filter_format == DataFormat.HWIO and \
                    op.input[1] in self._consts:
                producer = self._producer[op.input[0]]
                weight = self._consts[op.input[1]]
                if len(weight.dims) == 2 and self.is_after_fc(op) and \
                        len(producer.output_shape[0].dims) == 2 and \
                        weight.dims[0] == producer.output_shape[0].dims[1]:
                    six.print_('convert matmul to fc')
                    op.type = MaceOp.FullyConnected.name
                    weight_data = np.array(weight.float_data).reshape(
                        weight.dims)
                    weight.dims[:] = [1, 1] + list(weight_data.shape)
                    return True

        return False

    def update_float_op_data_type(self):
        if self._option.quantize:
            return

        print("update op with float data type")
        net = self._model
        data_type = self._option.data_type
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

        for output_node in self._option.output_nodes:
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
            elif op.type in MaceTransposableDataFormatOps:
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
                            if arg.i == 1:
                                arg.i = 3
                            elif arg.i == 2:
                                arg.i = 1
                            elif arg.i == 3:
                                arg.i = 2

                        producer = self._producer[op.input[0]]
                        input_shape = producer.output_shape[0].dims
                        if producer.type == MaceOp.FullyConnected.name and \
                                len(input_shape) == 2:
                            axis_arg = ConverterUtil.get_arg(
                                op, MaceKeyword.mace_axis_str)
                            if axis_arg.i == 1:
                                axis_arg.i = 3

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
                if op.input[i] in self.input_name_map:
                    op.input[i] = self.input_name_map[op.input[i]]
            for i in range(len(op.output)):
                if op.output[i] in self.output_name_map:
                    op.name = MaceKeyword.mace_output_node_name \
                              + '_' + op.name
                    new_output_name = self.output_name_map[op.output[i]]
                    self._quantize_activation_info[new_output_name] = \
                        self._quantize_activation_info[op.output[i]]
                    op.output[i] = new_output_name

            data_type_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_op_data_type_str)
            mace_check(data_type_arg, "Data type does not exist for %s(%s)"
                       % (op.name, op.type))
            if data_type_arg.i == mace_pb2.DT_FLOAT:
                data_type_arg.i = mace_pb2.DT_UINT8
            elif data_type_arg.i == mace_pb2.DT_UINT8:
                mace_check(op.type == MaceOp.Quantize.name
                           or op.type == MaceOp.Dequantize.name,
                           "Only Quantization ops support uint8, "
                           "but got %s(%s)" % (op.name, op.type))
            else:
                mace_check(op.type == MaceOp.Quantize.name,
                           "Quantization only support float ops, "
                           "but get %s(%s, %s)"
                           % (op.name, op.type,
                              mace_pb2.DataType.Name(data_type_arg.i)))

        for i, input_node in enumerate(self._option.input_nodes.values()):
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
                                        MaceOp.FullyConnected.name]\
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
                if self._option.device == DeviceType.CPU.value \
                       or self._option.device == DeviceType.APU.value:
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
                elif self._option.device == DeviceType.HEXAGON.value or \
                        self._option.device == DeviceType.HTA.value:
                    quantized_tensor = \
                        quantize_util.quantize_bias_for_hexagon(
                            tensor.float_data)
                else:
                    mace_check(False, "wrong device.")
                tensor.data_type = mace_pb2.DT_INT32
            else:
                non_zero = self._option.device == DeviceType.CPU.value
                quantized_tensor = quantize_util.quantize(tensor.float_data,
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
                                   MaceOp.FullyConnected.name]:
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
        if not self._option.quantize:
            return False

        # Quantize info from fixpoint fine tune
        print("Transform fake quantize")
        range_file = self._option.quantize_range_file
        if range_file:
            return

        net = self._model
        for op in net.op:
            if op.type == 'FakeQuantWithMinMaxVars' or \
                   op.type == 'FakeQuantWithMinMaxArgs':
                producer_op = self._producer[op.input[0]]
                minval = ConverterUtil.get_arg(op, 'min').f
                maxval = ConverterUtil.get_arg(op, 'max').f
                quantize_info = \
                    self.add_quantize_info(producer_op, minval, maxval)
                self._quantize_activation_info[op.input[0]] = quantize_info
                # for add -> fakequant pattern
                self._quantize_activation_info[op.output[0]] = quantize_info
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
        if range_file:
            print("Add quantize tensor range")
            with open(range_file) as f:
                for line in f:
                    tensor_name, minmax = line.split("@@")[:2]
                    min_val, max_val = [float(i) for i in
                                        minmax.strip().split(",")]
                    scale, zero, min_val, max_val = \
                        quantize_util.adjust_range(min_val, max_val,
                                                   self._option.device,
                                                   non_zero=False)
                    activation_info = mace_pb2.QuantizeActivationInfo()
                    activation_info.minval = min_val
                    activation_info.maxval = max_val
                    activation_info.scale = scale
                    activation_info.zero_point = zero
                    self._quantize_activation_info[tensor_name] = activation_info  # noqa

            for op in self._model.op:
                if op.name.find(MaceKeyword.mace_output_node_name) >= 0:
                    continue
                for output in op.output:
                    mace_check(output in self._quantize_activation_info,
                               "%s does not have quantize activation info"
                               % op)
                    op.quantize_info.extend([
                        self._quantize_activation_info[output]])

        if not self._option.quantize:
            return False

        print("Add default quantize info for input")
        for i, input_node in enumerate(self._option.input_nodes.values()):
            if input_node.name not in self._quantize_activation_info:
                print("Input range %s: %s" % (input_node.name,
                                              str(input_node.range)))
                new_input_name = self.input_name_map[input_node.name]
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

        print("Add default quantize info for ops like Pooling, Softmax")
        for op in self._model.op:
            if op.type in [MaceOp.Pooling.name,
                           MaceOp.Reduce.name,
                           MaceOp.Squeeze.name,
                           MaceOp.Reshape.name,
                           MaceOp.ResizeBilinear.name,
                           MaceOp.BatchToSpaceND.name,
                           MaceOp.SpaceToBatchND.name]:
                del op.quantize_info[:]
                producer_op = self._producer[op.input[0]]
                if producer_op.output[0] in self._option.input_nodes:
                    new_input_name = self.input_name_map[producer_op.output[0]]
                    self.copy_quantize_info(
                        op, self._quantize_activation_info[new_input_name])
                else:
                    self.copy_quantize_info(op, producer_op.quantize_info[0])
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
            elif op.type == MaceOp.Softmax.name:
                del op.quantize_info[:]
                quantize_info = \
                    self.add_quantize_info(op, 0.0, 1.0)
                self._quantize_activation_info[op.output[0]] = quantize_info
            elif (op.type == MaceOp.Eltwise.name
                  and not op.quantize_info
                  and len(op.input) == 2
                  and len(op.input[0]) not in self._consts
                  and len(op.input[1]) not in self._consts):
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
                else:
                    mace_check(False, "Quantized Elementwise only support:"
                                      " SUM and SUB now.")
                quantize_info = \
                    self.add_quantize_info(op, minval, maxval)
                self._quantize_activation_info[op.output[0]] = quantize_info

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
        arg.i = mace_pb2.GPU_IMAGE if self._option.cl_mem_type == "image"\
            else mace_pb2.GPU_BUFFER

    def delete_after_check_nodes(self):
        if self._option.check_nodes != self._option.output_nodes:
            mace_check(len(self._option.check_nodes) == 1,
                       "Only support one check node now.")
            check_node = None
            for i in six.moves.range(len(self._model.op)):
                if self._model.op[i].output[0] in self._option.check_nodes:
                    check_node = self._model.op[i]
                    del self._model.op[i+1:]
                    break
            mace_check(check_node is not None, "check node not found.")

    def transform_caffe_reshape_and_flatten(self):
        net = self._model
        for op in net.op:
            if op.type == MaceOp.Reshape.name and \
                    len(op.input) == 1:
                print("Transform Caffe Reshape")
                dims = []
                dim_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_dim_str)
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

    def fold_fc_reshape(self):
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
                if [0, 1, 2, 4, 3] == list(perm):
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
                        return False

                    # Change Transpose op to ChannelShuffle
                    op.type = MaceOp.ChannelShuffle.name
                    del op.arg[:]
                    group_arg = op.arg.add()
                    group_arg.name = MaceKeyword.mace_group_str
                    group_arg.i = op.output_shape[0].dims[4]
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
