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

import copy
import numpy as np
from enum import Enum
from operator import mul

from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.converter_tool.base_converter import EltwiseType
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import PaddingMode
from mace.python.tools.converter_tool.base_converter import PoolingType
from mace.python.tools.converter_tool.base_converter import ReduceType
from mace.python.tools.converter_tool.base_converter import DataFormat
from mace.python.tools.convert_util import mace_check
from mace.python.tools import graph_util


ApuSupportedOps = [
    'Concat',
    'Conv2D',
    'DepthwiseConv2d',
    'Eltwise',
    'Pooling',
    'ResizeBilinear',
    'Reshape',
    'Softmax',
    'Squeeze',
]

ApuOp = Enum('ApuOp', [(op, op) for op in ApuSupportedOps], type=str)


class ApuOps(object):
    def __init__(self):
        self.apu_ops = {
            MaceOp.Concat.name: ApuOp.Concat.name,
            MaceOp.Conv2D.name: ApuOp.Conv2D.name,
            MaceOp.DepthwiseConv2d.name: ApuOp.DepthwiseConv2d.name,
            MaceOp.Eltwise.name: ApuOp.Eltwise.name,
            MaceOp.Pooling.name: ApuOp.Pooling.name,
            MaceOp.ResizeBilinear.name: ApuOp.ResizeBilinear.name,
            MaceOp.Reshape.name: ApuOp.Reshape.name,
            MaceOp.Softmax.name: ApuOp.Softmax.name,
            MaceOp.Squeeze.name: ApuOp.Squeeze.name,
        }

    def has_op(self, op_name):
        return op_name in self.apu_ops

    def map_nn_op(self, op_name):
        if op_name not in self.apu_ops:
            raise Exception('Could not map nn op for: ', op_name)
        return self.apu_ops[op_name]


class ApuConverter(base_converter.ConverterInterface):
    def __init__(self, option, model, quantize_activation_info):
        self._option = option
        self._model = model
        self._apu_ops = ApuOps()

    def run(self):
        self.common_check()
        self.add_op_output_type()
        const_data_num_arg = self._model.arg.add()
        const_data_num_arg.name = MaceKeyword.mace_const_data_num_arg_str
        const_data_num_arg.i = len(self._model.tensors)
        self.convert_ops()
        self.add_node_id()
        return self._model

    def common_check(self):
        for op in self._model.op:
            mace_check(len(op.input) >= 1,
                       op.name + ': apu does not support op with 0 input')
            mace_check(len(op.output) == 1,
                       op.name + ': apu only support single output op')
            mace_check(len(op.output) == len(op.output_shape),
                       op.name + ': length of output and output_shape not'
                       ' match')
            mace_check(len(op.output_shape[0].dims) <= 4,
                       op.name + ': apu only support 1D~4D tensor')
            mace_check(len(op.output) == len(op.quantize_info),
                       op.name + ': length of output and quantize_info not'
                       ' match')
            data_format = ConverterUtil.data_format(op)
            if data_format is not None and len(op.output_shape[0].dims) == 4:
                mace_check(data_format == DataFormat.NHWC,
                           op.name + ': apu only support 4D tensor with NHWC'
                           ' format')
        for tensor in self._model.tensors:
            mace_check(len(tensor.dims) <= 4,
                       tensor.name + ': apu only support 1D~4D tensor')
        for input_info in self._model.input_info:
            mace_check(len(input_info.dims) <= 4,
                       input_info.name + ': apu only support 1D~4D tensor')
            mace_check(input_info.data_type == mace_pb2.DT_UINT8
                       or input_info.data_type == mace_pb2.DT_INT32,
                       input_info.name + ': apu not support float input')
            if len(input_info.dims) == 4:
                mace_check(input_info.data_format == DataFormat.NHWC.value,
                           input_info.name + ': apu only support 4D tensor'
                           ' with NHWC format')

    def convert_ops(self):
        print("Convert mace graph to apu.")
        for op in self._model.op:
            if not self._apu_ops.has_op(op.type):
                raise Exception('Unsupported op: ', op)

            if op.type == MaceOp.Conv2D.name \
                    or op.type == MaceOp.DepthwiseConv2d.name:
                mace_check(len(op.input) == 3,
                           op.name + ': apu only support ' + op.type + ' op'
                           ' with 3 input')
                self.add_size_tensor_from_arg(
                    op, MaceKeyword.mace_strides_str)
                self.add_padding_tensor_from_arg(op)
                self.add_size_tensor_from_arg(
                    op, MaceKeyword.mace_dilations_str)
                if op.type == MaceOp.DepthwiseConv2d.name:
                    multiplier = self._model.tensors.add()
                    multiplier.name = op.name + '/multiplier:0'
                    multiplier.data_type = mace_pb2.DT_INT32
                    multiplier.dims.extend([1])
                    for tensor in self._model.tensors:
                        if tensor.name == op.input[1]:
                            multiplier.int32_data.extend([tensor.dims[0]])
                            break
                    op.input.extend([multiplier.name])
            elif op.type == MaceOp.Eltwise.name:
                mace_check(len(op.input) == 2,
                           op.name + ': apu only support eltwise op with 2'
                           ' input')
                eltwise_type = ConverterUtil.get_arg(
                               op, MaceKeyword.mace_element_type_str).i
                mace_check(eltwise_type == EltwiseType.SUM.value,
                           op.name + ': apu only support eltwise type SUM')
            elif op.type == MaceOp.Pooling.name:
                mace_check(len(op.input) == 1,
                           op.name + ': apu only support pooling op with 1'
                           ' input')
                pooling_type_arg = ConverterUtil.get_arg(
                                   op, MaceKeyword.mace_pooling_type_str)
                mace_check(PoolingType(pooling_type_arg.i) == PoolingType.AVG,
                           op.name + ': apu only support pooling type AVG')
                self.add_padding_tensor_from_arg(op)
                self.add_size_tensor_from_arg(
                    op, MaceKeyword.mace_strides_str)
                self.add_size_tensor_from_arg(op, MaceKeyword.mace_kernel_str)
            elif op.type == MaceOp.Concat.name:
                self.add_int_tensor_from_arg(op, MaceKeyword.mace_axis_str)
            elif op.type == MaceOp.ResizeBilinear.name:
                mace_check(len(op.input) == 1,
                           op.name + ': apu only support resize bilinear op'
                           ' with 1 input')
                self.add_int_tensor_from_arg(
                    op, MaceKeyword.mace_align_corners_str)
            elif op.type == MaceOp.Reshape.name:
                mace_check(len(op.input) == 1 or len(op.input) == 2,
                           op.name + ': apu only support reshape op with 1 or'
                           ' 2 input')
            elif op.type == MaceOp.Softmax.name:
                mace_check(len(op.input) == 1,
                           op.name + ': apu only support softmax op with 1'
                           ' input')
                beta_value_tensor = self._model.tensors.add()
                beta_value_tensor.name = op.name + '/beta:0'
                beta_value_tensor.data_type = mace_pb2.DT_FLOAT
                beta_value_tensor.dims.extend([1])
                beta_value_tensor.float_data.extend([1.0])
                op.input.extend([beta_value_tensor.name])
            elif op.type == MaceOp.Squeeze.name:
                mace_check(len(op.input) == 1,
                           op.name + ': apu only support squeeze op with 1'
                           ' input')
                axis_tensor_arg = ConverterUtil.get_arg(
                                      op, MaceKeyword.mace_axis_str)
                axis_tensor = self._model.tensors.add()
                axis_tensor.name = op.name + '/axis:0'
                axis_tensor.data_type = mace_pb2.DT_INT32
                axis_tensor.dims.extend([len(axis_tensor_arg.ints)])
                axis_tensor.int32_data.extend(axis_tensor_arg.ints)
                op.input.extend([axis_tensor.name])

            op.type = self._apu_ops.map_nn_op(op.type)

    def add_op_output_type(self):
        type_map = {}
        for input_info in self._model.input_info:
            mace_check(input_info.data_type is not None,
                       input_info.name + ' does not have data_type')
            type_map[input_info.name] = input_info.data_type

        for op in self._model.op:
            if len(op.output_type) >= 1:
                print([op.name, len(op.output), len(op.output_type)])
                type_map[op.output[0]] = op.output_type[0]
                continue
            mace_check(op.input[0] in type_map,
                       op.input[0] + ' not in type_map')
            op.output_type.extend([type_map[op.input[0]]])
            type_map[op.output[0]] = op.output_type[0]

        for op in self._model.op:
            mace_check(len(op.output) == len(op.output_type),
                       op.name + ': length of output and output_type not'
                       ' match')
            mace_check(op.output_type[0] == mace_pb2.DT_UINT8
                       or op.output_type[0] == mace_pb2.DT_INT32,
                       op.name + ': apu only support quantize node')

    def add_node_id(self):
        node_id_counter = 0
        node_id_map = {}
        for tensor in self._model.tensors:
            tensor.node_id = node_id_counter
            node_id_counter += 1
            node_id_map[tensor.name] = tensor.node_id
        for input_info in self._model.input_info:
            input_info.node_id = node_id_counter
            node_id_counter += 1
            node_id_map[input_info.name] = input_info.node_id
        for op in self._model.op:
            op.node_id = node_id_counter
            node_id_counter += 1
            node_id_map[op.output[0]] = op.node_id

        for op in self._model.op:
            del op.node_input[:]
            for input_tensor in op.input:
                node_input = op.node_input.add()
                node_input.node_id = node_id_map[input_tensor]
        for output_info in self._model.output_info:
            output_info.node_id = node_id_map[output_info.name]

    def add_padding_tensor_from_arg(self, op):
        padding_value_arg = ConverterUtil.get_arg(
                            op, MaceKeyword.mace_padding_values_str)
        mace_check(len(padding_value_arg.ints) == 4,
                   op.name + ': padding value does not have size 4')
        padding_value_tensor = self._model.tensors.add()
        padding_value_tensor.name = op.name + '/padding:0'
        padding_value_tensor.data_type = mace_pb2.DT_INT32
        padding_value_tensor.dims.extend([4])
        padding_value_tensor.int32_data.extend(padding_value_arg.ints)
        op.input.extend([padding_value_tensor.name])

    def add_size_tensor_from_arg(self, op, keyword):
        size_value_arg = ConverterUtil.get_arg(op, keyword)
        mace_check(len(size_value_arg.ints) == 2,
                   op.name + ': ' + keyword + ' value does not have size 2')
        size_value_tensor = self._model.tensors.add()
        size_value_tensor.name = op.name + '/' + keyword + ':0'
        size_value_tensor.data_type = mace_pb2.DT_INT32
        size_value_tensor.dims.extend([2])
        size_value_tensor.int32_data.extend(size_value_arg.ints)
        op.input.extend([size_value_tensor.name])

    def add_int_tensor_from_arg(self, op, keyword):
        int_value_arg = ConverterUtil.get_arg(op, keyword)
        mace_check(int_value_arg.i is not None,
                   op.name + ': ' + keyword + ' value i should not be None')
        int_value_tensor = self._model.tensors.add()
        int_value_tensor.name = op.name + '/' + keyword + ':0'
        int_value_tensor.data_type = mace_pb2.DT_INT32
        int_value_tensor.dims.extend([1])
        int_value_tensor.int32_data.extend([int_value_arg.i])
        op.input.extend([int_value_tensor.name])
