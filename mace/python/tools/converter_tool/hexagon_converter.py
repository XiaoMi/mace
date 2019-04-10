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
from mace.python.tools.converter_tool.base_converter import DeviceType
from mace.python.tools.converter_tool.base_converter import EltwiseType
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import PaddingMode
from mace.python.tools.converter_tool.base_converter import PoolingType
from mace.python.tools.converter_tool.base_converter import ReduceType
from mace.python.tools.convert_util import mace_check
from mace.python.tools import graph_util

from six.moves import reduce


HexagonSupportedOps = [
    'BatchToSpaceND_8',
    'DepthwiseSupernode_8x8p32to8',
    'DequantizeOUTPUT_8tof',
    'INPUT',
    'OUTPUT',
    'QuantizedAdd_8p8to8',
    'QuantizedAvgPool_8',
    'QuantizedConcat_8',
    'QuantizedMaxPool_8',
    'QuantizedResizeBilinear_8',
    'QuantizedSoftmax_8',
    'QuantizeINPUT_f_to_8',
    'SpaceToBatchND_8',
    'Supernode_8x8p32to8',
    'Nop',
]

HexagonOp = Enum('HexagonOp', [(op, op) for op in HexagonSupportedOps],
                 type=str)


class HexagonOps(object):
    def __init__(self):
        self.hexagon_ops = {
            MaceOp.BatchToSpaceND.name: HexagonOp.BatchToSpaceND_8.name,
            MaceOp.Concat.name: HexagonOp.QuantizedConcat_8.name,
            MaceOp.Conv2D.name: HexagonOp.Supernode_8x8p32to8.name,
            MaceOp.DepthwiseConv2d.name:
                HexagonOp.DepthwiseSupernode_8x8p32to8.name,
            MaceOp.Dequantize.name: HexagonOp.DequantizeOUTPUT_8tof.name,
            MaceOp.Eltwise.name: [HexagonOp.QuantizedAdd_8p8to8],
            MaceOp.Identity.name: HexagonOp.Nop.name,
            MaceOp.Quantize.name: HexagonOp.QuantizeINPUT_f_to_8.name,
            MaceOp.Pooling.name: [HexagonOp.QuantizedAvgPool_8.name,
                                  HexagonOp.QuantizedMaxPool_8.name],
            MaceOp.Reduce.name: HexagonOp.QuantizedAvgPool_8.name,
            MaceOp.ResizeBilinear.name:
                HexagonOp.QuantizedResizeBilinear_8.name,
            MaceOp.SpaceToBatchND.name: HexagonOp.SpaceToBatchND_8.name,
            MaceOp.Softmax.name: HexagonOp.QuantizedSoftmax_8.name,
        }

    def has_op(self, tf_op):
        return tf_op in self.hexagon_ops

    def map_nn_op(self, tf_op):
        if tf_op not in self.hexagon_ops:
            raise Exception('Could not map nn op for: ', tf_op)
        return self.hexagon_ops[tf_op]


padding_mode = {
    PaddingMode.NA: 0,
    PaddingMode.SAME: 1,
    PaddingMode.VALID: 2
}


def get_tensor_name_from_op(op_name, port):
    return op_name + ':' + str(port)


def get_op_and_port_from_tensor(tensor_name):
    if ':' in tensor_name:
        op, port = tensor_name.split(':')
        port = int(port)
    else:
        op = tensor_name
        port = 0
    return op, port


def normalize_name(name):
    return name.replace(':', '_')


class HexagonConverter(base_converter.ConverterInterface):
    def __init__(self, option, model, quantize_activation_info):
        self._option = option
        self._model = model
        self._hexagon_ops = HexagonOps()
        self._consts = {}
        self._quantize_activation_info = quantize_activation_info

    def run(self):
        if self._option.device == DeviceType.HTA.value:
            mace_check(len(self._option.input_nodes) == 1
                       and len(self._option.output_nodes) == 1,
                       'hta only support single input and output')

        for tensor in self._model.tensors:
            self._consts[tensor.name] = tensor

        # convert op node
        self.convert_ops()

        self.convert_input_output_node()

        self.add_node_id()

        return self._model

    def convert_ops(self):
        print("Convert mace graph to hexagon.")
        for op in self._model.op:
            if not self._hexagon_ops.has_op(op.type):
                raise Exception('Unsupported op: ', op)
            for i in range(len(op.input)):
                if ':' not in op.input[i]:
                    node_name = op.input[i]
                    op.input[i] += ':0'
                    if node_name in self._quantize_activation_info:
                        self._quantize_activation_info[op.input[i]] = \
                            self._quantize_activation_info[node_name]

            if op.type == MaceOp.Conv2D.name \
                    or op.type == MaceOp.DepthwiseConv2d.name:
                channels = op.output_shape[0].dims[3]

                if len(op.input) < 3:
                    print('Supernode requires biasadd, we add it.')
                    bias_data = np.zeros(channels, dtype=int)
                    bias_tensor = self._model.tensors.add()
                    bias_tensor.data_type = mace_pb2.DT_INT32
                    bias_tensor.dims.extend([channels])
                    bias_tensor.int32_data.extend(bias_data)
                    bias_tensor.minval = 0
                    bias_tensor.maxval = 0
                    bias_tensor.name = op.name + "/bias:0"
                    bias = bias_tensor.name
                    self._consts[bias] = bias_tensor
                else:
                    bias = op.input.pop()

                self.add_min_max_const_node(op, op.input[0])
                self.add_min_max_const_node(op, op.input[1])
                strides_arg = ConverterUtil.get_arg(op, 'strides')
                mace_check(strides_arg is not None,
                           "Missing strides of Conv or Depthwise Conv.")
                strides = self.add_shape_const_node(
                    op, [1, strides_arg.ints[0], strides_arg.ints[1], 1],
                    MaceKeyword.mace_strides_str)
                op.input.extend([strides, bias])
                self.add_min_max_const_node(op, bias)
                self.add_min_max_const_node(
                    op, op.output[0], True, True, False)
            elif op.type == MaceOp.Eltwise.name:
                self.add_min_max_const_node(op, op.input[0])
                self.add_min_max_const_node(op, op.input[1])
                self.add_min_max_const_node(
                    op, op.output[0], True, True, False)
            elif op.type == MaceOp.BatchToSpaceND.name \
                    or op.type == MaceOp.SpaceToBatchND.name:
                strides_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_space_batch_block_shape_str)
                strides_tensor = self._model.tensors.add()
                strides_tensor.name = op.name + '/strides:0'
                strides_tensor.data_type = mace_pb2.DT_INT32
                strides_tensor.dims.extend([1, 1, 1, len(strides_arg.ints)])
                strides_tensor.int32_data.extend(strides_arg.ints)
                if op.type == MaceOp.BatchToSpaceND.name:
                    pad_arg = ConverterUtil.get_arg(
                        op, MaceKeyword.mace_batch_to_space_crops_str)
                else:
                    pad_arg = ConverterUtil.get_arg(
                        op, MaceKeyword.mace_paddings_str)
                pad_tensor = self._model.tensors.add()
                pad_tensor.name = op.name + '/pad:0'
                pad_tensor.data_type = mace_pb2.DT_INT32
                pad_tensor.dims.extend([1, 1, len(pad_arg.ints) / 2, 2])
                pad_tensor.int32_data.extend(pad_arg.ints)
                op.input.extend([strides_tensor.name, pad_tensor.name])
                self.add_min_max_const_node(op, op.input[0])
            elif op.type == MaceOp.Pooling.name:
                self.add_min_max_const_node(op, op.input[0])
                window_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_kernel_str)
                window_tensor = self._model.tensors.add()
                window_tensor.name = op.name + '/window:0'
                window_tensor.data_type = mace_pb2.DT_INT32
                window_tensor.dims.extend(
                    [1, window_arg.ints[0], window_arg.ints[1], 1])
                strides_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_strides_str)
                strides_tensor = self._model.tensors.add()
                strides_tensor.name = op.name + '/strides:0'
                strides_tensor.data_type = mace_pb2.DT_INT32
                strides_tensor.dims.extend(
                    [1, strides_arg.ints[0], strides_arg.ints[1], 1])
                op.input.extend([window_tensor.name, strides_tensor.name])
            elif op.type == MaceOp.Reduce.name:
                self.add_min_max_const_node(op, op.input[0])
                reduce_type_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_reduce_type_str)
                mace_check(reduce_type_arg.i == ReduceType.MEAN.value,
                           "Hexagon Reduce only supports Mean now.")
                keep_dims_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_keepdims_str)
                mace_check(keep_dims_arg.i == 1,
                           "Hexagon Reduce Mean only supports keep dims now.")
                axis_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str)
                mace_check(1 <= len(axis_arg.ints) <= 2,
                           "Hexagon Reduce Mean only supports spatial now.")
                for i in axis_arg.ints:
                    mace_check(1 <= i <= 2,
                               "Hexagon Reduce Mean only supports spatial now")
                producer_op_name, _ = get_op_and_port_from_tensor(op.input[0])
                input_dims = None
                for producer_op in self._model.op:
                    if producer_op.name == producer_op_name:
                        input_dims = producer_op.output_shape[0].dims
                        break
                mace_check(input_dims is not None, "Missing input shape.")
                window_tensor = self._model.tensors.add()
                window_tensor.name = op.name + '/window:0'
                window_tensor.data_type = mace_pb2.DT_INT32
                if len(axis_arg.ints) == 1:
                    dim1, dim2 = (input_dims[1], 1) \
                        if axis_arg.ints[0] == 1 else (1, input_dims[2])
                else:
                    dim1, dim2 = input_dims[1], input_dims[2]
                window_tensor.dims.extend([1, dim1, dim2, 1])
                strides_tensor = self._model.tensors.add()
                strides_tensor.name = op.name + '/strides:0'
                strides_tensor.data_type = mace_pb2.DT_INT32
                strides_tensor.dims.extend([1, dim1, dim2, 1])
                op.input.extend([window_tensor.name, strides_tensor.name])
            elif op.type == MaceOp.ResizeBilinear.name:
                newdim_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_resize_size_str)
                newdim_tensor = self._model.tensors.add()
                newdim_tensor.name = op.name + '/newdim:0'
                newdim_tensor.data_type = mace_pb2.DT_INT32
                newdim_tensor.dims.extend([len(newdim_arg.ints)])
                newdim_tensor.int32_data.extend(newdim_arg.ints)
                op.input.extend([newdim_tensor.name])
                self.add_min_max_const_node(op, op.input[0])
                align_corners_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_align_corners_str)
                align_corners_tensor = self._model.tensors.add()
                align_corners_tensor.name = op.name + '/align_corners:0'
                align_corners_tensor.data_type = mace_pb2.DT_INT32
                align_corners_tensor.dims.extend([1])
                align_corners_tensor.int32_data.extend([align_corners_arg.i])
                op.input.extend([align_corners_tensor.name])
            elif op.type == MaceOp.Concat.name:
                inputs = copy.deepcopy(op.input)
                for ipt in inputs:
                    self.add_min_max_const_node(op, ipt, True, False)
                for ipt in inputs:
                    self.add_min_max_const_node(op, ipt, False, True)
                dim_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_axis_str)
                dim_tensor = self._model.tensors.add()
                dim_tensor.name = op.name + '/dim:0'
                dim_tensor.data_type = mace_pb2.DT_INT32
                dim_tensor.dims.extend([1])
                dim_tensor.int32_data.extend([dim_arg.i])
                op.input.insert(0, dim_tensor.name)
            elif op.type in [MaceOp.Softmax.name,
                             MaceOp.Dequantize.name]:
                self.add_min_max_const_node(op, op.input[0])

            if op.type != MaceOp.Dequantize.name:
                min_output_shape = op.output_shape.add()
                min_output_shape.dims.extend([1])
                max_output_shape = op.output_shape.add()
                max_output_shape.dims.extend([1])
                op.output_type.extend(
                    [mace_pb2.DT_UINT8, mace_pb2.DT_FLOAT, mace_pb2.DT_FLOAT])
            for i in range(len(op.output_shape)):
                out_max_byte_size = reduce(mul, op.output_shape[i].dims)
                if op.output_type[i] == mace_pb2.DT_FLOAT:
                    out_max_byte_size *= 4
                op.out_max_byte_size.extend([out_max_byte_size])

            op.padding = padding_mode[PaddingMode.NA]
            arg = ConverterUtil.get_arg(op, MaceKeyword.mace_padding_str)
            if arg is not None:
                op.padding = padding_mode[PaddingMode(arg.i)]

            if (op.type == MaceOp.Eltwise.name
                    and ConverterUtil.get_arg(
                        op, MaceKeyword.mace_element_type_str).i
                    == EltwiseType.SUM.value):
                op.type = HexagonOp.QuantizedAdd_8p8to8.name
            elif op.type == MaceOp.Pooling.name:
                pooling_type_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_pooling_type_str)
                if PoolingType(pooling_type_arg.i) == PoolingType.AVG:
                    op.type = HexagonOp.QuantizedAvgPool_8.name
                else:
                    op.type = HexagonOp.QuantizedMaxPool_8.name
            else:
                op.type = self._hexagon_ops.map_nn_op(op.type)

    def add_const_node(self, name, val):
        if name not in self._consts:
            tensor = self._model.tensors.add()
            self._consts[name] = tensor
            tensor.name = name
            tensor.data_type = mace_pb2.DT_FLOAT
            tensor.dims.extend([1])
            tensor.float_data.extend([val])

    def add_min_max_const_node(
            self, this_op, tensor_name, add_min=True, add_max=True,
            diff_port=True):
        op, port = get_op_and_port_from_tensor(tensor_name)
        mace_check(port == 0, 'port should be 0 to add min max tensor then.')
        if tensor_name in self._quantize_activation_info:
            quantize_info = self._quantize_activation_info[tensor_name]
            minval = quantize_info.minval
            maxval = quantize_info.maxval
            is_activation = True
        elif tensor_name in self._consts:
            tensor = self._consts[tensor_name]
            minval = tensor.minval
            maxval = tensor.maxval
            is_activation = False
        else:
            raise Exception('Quantize info not found: ', tensor_name)

        if add_min:
            if is_activation and diff_port:
                min_tensor_name = op + ':1'
            else:
                min_tensor_name = op + '_min:0'
                self.add_const_node(min_tensor_name, minval)
            this_op.input.extend([min_tensor_name])
        if add_max:
            if is_activation and diff_port:
                max_tensor_name = op + ':2'
            else:
                max_tensor_name = op + '_max:0'
                self.add_const_node(max_tensor_name, maxval)
            this_op.input.extend([max_tensor_name])

    def add_shape_const_node(self, op, values, name):
        tensor = self._model.tensors.add()
        node_name = op.name + '/' + name
        tensor.name = node_name + ':0'
        tensor.data_type = mace_pb2.DT_INT32
        tensor.dims.extend(values)
        return tensor.name

    def add_constant_min_max_for_first_op(self, op):
        minval = self._quantize_activation_info[op.input[0]].minval
        maxval = self._quantize_activation_info[op.input[0]].maxval
        input_op, _ = get_op_and_port_from_tensor(op.input[0])
        input_min = input_op + '_min:0'
        input_max = input_op + '_max:0'
        self.add_const_node(input_min, minval)
        self.add_const_node(input_max, maxval)
        for i in range(len(op.input)):
            if op.input[i] == input_op + ':1':
                op.input[i] = input_min
            elif op.input[i] == input_op + ':2':
                op.input[i] = input_max

    def convert_input_output_node(self):
        quantize_input_op = self._model.op[0]
        mace_check(
            quantize_input_op.type == HexagonOp.QuantizeINPUT_f_to_8.name,
            "Not started with Quantize op.")
        del quantize_input_op.input[:]

        dequantize_output_op = self._model.op[-1]
        mace_check(dequantize_output_op.type
                   == HexagonOp.DequantizeOUTPUT_8tof.name,
                   "Not ended with Dequantize op.")
        dequantize_input = [input for input in dequantize_output_op.input]
        del dequantize_output_op.input[:]
        del dequantize_output_op.output_shape[:]
        del dequantize_output_op.output_type[:]
        del dequantize_output_op.out_max_byte_size[:]

        index = 1
        while index < len(self._model.op) - 1:
            op = self._model.op[index]
            if op.type == HexagonOp.QuantizeINPUT_f_to_8.name:
                quantize_input_op.output.extend(op.output)
                quantize_input_op.output_shape.extend(op.output_shape)
                quantize_input_op.output_type.extend(op.output_type)
                quantize_input_op.out_max_byte_size.extend(
                    op.out_max_byte_size)
                del self._model.op[index]

            elif op.type == HexagonOp.DequantizeOUTPUT_8tof.name:
                dequantize_output_op.input.extend(op.input)
                del self._model.op[index]

            index += 1
        # input order matters
        dequantize_output_op.input.extend(dequantize_input)

        if self._option.device == DeviceType.HTA.value:
            # replace QuantizeINPUT_f_to_8 with INPUT
            quantize_input_op.type = HexagonOp.INPUT.name
            del quantize_input_op.output_shape[1:]
            del quantize_input_op.output_type[1:]
            del quantize_input_op.out_max_byte_size[1:]

            # replace first op's input min max with constant
            self.add_constant_min_max_for_first_op(self._model.op[1])

            # replace DequantizeOUTPUT_8tof with OUTPUT
            dequantize_output_op.type = HexagonOp.OUTPUT.name
            del dequantize_output_op.input[1:]

    def add_node_id(self):
        node_id_counter = 0
        node_id_map = {}
        for tensor in self._model.tensors:
            tensor.node_id = node_id_counter
            node_id_counter += 1
            tensor_op, port = get_op_and_port_from_tensor(tensor.name)
            node_id_map[tensor_op] = tensor.node_id

        print("Hexagon op:")
        index = 0
        for op in self._model.op:
            op.node_id = node_id_counter
            if op.type not in [HexagonOp.QuantizeINPUT_f_to_8,
                               HexagonOp.DequantizeOUTPUT_8tof.name]:
                index_str = str(index)
                index += 1
            else:
                index_str = ''
            print('Op: %s (%s, node_id:%d, index:%s)' %
                  (op.name, op.type, op.node_id, index_str))
            node_id_counter += 1
            node_id_map[op.name] = op.node_id
            for ipt in op.input:
                op_name, port = get_op_and_port_from_tensor(ipt)
                node_id = node_id_map[op_name]
                node_input = op.node_input.add()
                node_input.node_id = node_id
                node_input.output_port = int(port)
