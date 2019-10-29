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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import copy
import numpy as np
from enum import Enum
from operator import mul
from functools import reduce

from py_proto import mace_pb2
from transform import base_converter
from transform.base_converter import ConverterUtil
from transform.base_converter import DeviceType
from transform.base_converter import EltwiseType
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp
from transform.base_converter import PaddingMode
from transform.base_converter import PoolingType
from transform.base_converter import ReduceType
from utils.util import mace_check


HexagonSupportedOps = [
    'BatchToSpaceND_8',
    'DepthToSpace_8',
    'DepthwiseSupernode_8x8p32to8',
    'DequantizeOUTPUT_8tof',
    'INPUT',
    'OUTPUT',
    'QuantizedAdd_8p8to8',
    'QuantizedAvgPool_8',
    'QuantizedConcat_8',
    'QuantizedMaxPool_8',
    'QuantizedMul_8x8to8',
    'QuantizedResizeBilinear_8',
    'QuantizedSoftmax_8',
    'QuantizedSub_8p8to8',
    'QuantizeINPUT_f_to_8',
    'SpaceToBatchND_8',
    'SpaceToDepth_8',
    'Supernode_8x8p32to8',
    'Nop',
]

HexagonOp = Enum('HexagonOp', [(op, op) for op in HexagonSupportedOps],
                 type=str)

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
        self._consts = {}
        self._quantize_activation_info = quantize_activation_info
        self._op_converters = {
            MaceOp.BatchToSpaceND.name: self.convert_batchspace,
            MaceOp.Concat.name: self.convert_concat,
            MaceOp.Conv2D.name: self.convert_conv2d,
            MaceOp.DepthToSpace.name: self.convert_depthspace,
            MaceOp.DepthwiseConv2d.name: self.convert_conv2d,
            MaceOp.Dequantize.name: self.convert_dequantize,
            MaceOp.Eltwise.name: self.convert_elementwise,
            MaceOp.Pooling.name: self.convert_pooling,
            MaceOp.Quantize.name: self.convert_quantize,
            MaceOp.Reduce.name: self.convert_reduce,
            MaceOp.ResizeBilinear.name: self.convert_resizebilinear,
            MaceOp.Softmax.name: self.convert_softmax,
            MaceOp.SpaceToBatchND.name: self.convert_batchspace,
            MaceOp.SpaceToDepth.name: self.convert_depthspace,
        }

    def run(self):
        if self._option.device == DeviceType.HTA.value:
            mace_check(len(self._option.input_nodes) == 1
                       and len(self._option.output_nodes) == 1,
                       'hta only support single input and output')

        for tensor in self._model.tensors:
            self._consts[tensor.name] = tensor

        # convert op node
        self.convert_ops()

        model_inputs = self.convert_input_output_node()

        self.add_node_id(model_inputs)

        return self._model

    def add_port_for_tensors(self,  tensors):
        for i in range(len(tensors)):
            if ':' not in tensors[i]:
                node_name = tensors[i]
                tensors[i] += ':0'
                if node_name in self._quantize_activation_info:
                    self._quantize_activation_info[tensors[i]] = \
                        self._quantize_activation_info[node_name]

    def add_const_node(self, name, val):
        if name not in self._consts:
            tensor = self._model.tensors.add()
            self._consts[name] = tensor
            tensor.name = name
            tensor.data_type = mace_pb2.DT_FLOAT
            tensor.dims.extend([1])
            tensor.float_data.extend([val])

    def add_arg_const_node(self, op, name, dims, data=None, insert_index=None):
        arg_tensor = self._model.tensors.add()
        arg_tensor.name = op.name + name
        arg_tensor.data_type = mace_pb2.DT_INT32
        arg_tensor.dims.extend(dims)
        if data:
            arg_tensor.int32_data.extend(data)
        if insert_index is None:
            op.input.append(arg_tensor.name)
        else:
            op.input.insert(insert_index, arg_tensor.name)

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
        first_quantize_input_op = copy.deepcopy(quantize_input_op)
        del quantize_input_op.input[:]
        del quantize_input_op.output[:]
        del quantize_input_op.output_shape[:]
        del quantize_input_op.output_type[:]
        del quantize_input_op.out_max_byte_size[:]

        dequantize_output_op = self._model.op[-1]
        mace_check(dequantize_output_op.type
                   == HexagonOp.DequantizeOUTPUT_8tof.name,
                   "Not ended with Dequantize op.")
        last_dequantize_output_op = copy.deepcopy(dequantize_output_op)
        del dequantize_output_op.input[:]
        del dequantize_output_op.output[:]
        del dequantize_output_op.output_shape[:]
        del dequantize_output_op.output_type[:]
        del dequantize_output_op.out_max_byte_size[:]

        # Combine multiple inputs/outputs to one hexagon input/output node,
        # in input_info/output_info order
        ops = {}
        for op in self._model.op:
            ops[op.name] = op
        for input_node in self._option.input_nodes.values():
            op_name = normalize_name(
                MaceKeyword.mace_input_node_name + '_' + input_node.name)
            if op_name == first_quantize_input_op.name:
                op = first_quantize_input_op
                quantize_input_op.name = MaceKeyword.mace_input_node_name
            else:
                op = ops[op_name]
            mace_check(op.type == HexagonOp.QuantizeINPUT_f_to_8.name,
                       "input node type is: %s" % op.type)
            quantize_input_op.output.extend(op.output)
            quantize_input_op.output_shape.extend(op.output_shape)
            quantize_input_op.output_type.extend(op.output_type)
            quantize_input_op.out_max_byte_size.extend(
                op.out_max_byte_size)
        for output_node in self._option.check_nodes.values():
            op_name = normalize_name(output_node.name)
            op = last_dequantize_output_op \
                if op_name == last_dequantize_output_op.name else ops[op_name]
            mace_check(op.type == HexagonOp.DequantizeOUTPUT_8tof.name,
                       "output node type is: %s" % op.type)
            dequantize_output_op.input.extend(op.input)

        # Delete redundant inputs/outputs nodes
        index = 1
        while index < len(self._model.op) - 1:
            op = self._model.op[index]
            if op.type == HexagonOp.QuantizeINPUT_f_to_8.name \
                    or op.type == HexagonOp.DequantizeOUTPUT_8tof.name:
                del self._model.op[index]
            else:
                index += 1

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

        return quantize_input_op.output

    def add_node_id(self, model_inputs):
        node_id_counter = 0
        node_id_map = {}
        for tensor in self._model.tensors:
            tensor.node_id = node_id_counter
            node_id_counter += 1
            node_id_map[tensor.name] = tensor.node_id

        print("Hexagon op:")
        index = 0
        for op in self._model.op:
            op.node_id = node_id_counter
            node_id_counter += 1
            for output in op.output:
                node_id_map[output] = op.node_id
            if op.type not in [HexagonOp.QuantizeINPUT_f_to_8,
                               HexagonOp.DequantizeOUTPUT_8tof.name]:
                index_str = str(index)
                index += 1
            else:
                index_str = ''
            print('Op: %s (%s, node_id:%d, index:%s)' %
                  (op.name, op.type, op.node_id, index_str))
            for ipt in op.input:
                op_name, port = get_op_and_port_from_tensor(ipt)
                tensor_name = ipt if port == 0 else op_name + ':0'
                node_id = node_id_map[tensor_name]
                node_input = op.node_input.add()
                node_input.node_id = node_id
                if tensor_name in model_inputs:
                    for i in range(len(model_inputs)):
                        if model_inputs[i] == tensor_name:
                            port += i * 3
                node_input.output_port = port

    def convert_ops(self):
        print("Convert mace graph to hexagon.")
        for op in self._model.op:
            mace_check(op.type in self._op_converters,
                       "Mace Hexagon does not support op type %s yet"
                       % op.type)
            self.pre_convert(op)
            self._op_converters[op.type](op)
            self.post_convert(op)

    def pre_convert(self, op):
        self.add_port_for_tensors(op.input)
        self.add_port_for_tensors(op.output)

    def post_convert(self, op):
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

    def convert_batchspace(self, op):
        strides_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_space_batch_block_shape_str)
        self.add_arg_const_node(
            op, '/strides:0', [1, 1, 1, len(strides_arg.ints)],
            strides_arg.ints)

        if op.type == MaceOp.BatchToSpaceND.name:
            pad_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_batch_to_space_crops_str)
        else:
            pad_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_paddings_str)
        self.add_arg_const_node(
            op, '/pad:0', [1, 1, len(pad_arg.ints) // 2, 2], pad_arg.ints)

        self.add_min_max_const_node(op, op.input[0])

        if op.type == MaceOp.BatchToSpaceND.name:
            op.type = HexagonOp.BatchToSpaceND_8.name
        else:
            op.type = HexagonOp.SpaceToBatchND_8.name

    def convert_concat(self, op):
        inputs = copy.deepcopy(op.input)
        for ipt in inputs:
            self.add_min_max_const_node(op, ipt, True, False)
        for ipt in inputs:
            self.add_min_max_const_node(op, ipt, False, True)

        dim_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_axis_str)
        self.add_arg_const_node(op, '/dim:0', [1], [dim_arg.i], 0)

        op.type = HexagonOp.QuantizedConcat_8.name

    def convert_conv2d(self, op):
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
        self.add_arg_const_node(
            op, '/strides:0', [1, strides_arg.ints[0], strides_arg.ints[1], 1])

        op.input.append(bias)
        self.add_min_max_const_node(op, bias)
        self.add_min_max_const_node(
            op, op.output[0], True, True, False)

        if op.type == MaceOp.DepthwiseConv2d.name:
            op.type = HexagonOp.DepthwiseSupernode_8x8p32to8.name
        else:
            op.type = HexagonOp.Supernode_8x8p32to8.name

    def convert_depthspace(self, op):
        size_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_space_depth_block_size_str)
        self.add_arg_const_node(op, '/block_size:0', [1], [size_arg.i])

        self.add_min_max_const_node(op, op.input[0])

        if op.type == MaceOp.DepthToSpace.name:
            op.type = HexagonOp.DepthToSpace_8.name
        else:
            op.type = HexagonOp.SpaceToDepth_8.name

    def convert_dequantize(self, op):
        self.add_min_max_const_node(op, op.input[0])

        op.type = HexagonOp.DequantizeOUTPUT_8tof.name

    def convert_elementwise(self, op):
        self.add_min_max_const_node(op, op.input[0])
        self.add_min_max_const_node(op, op.input[1])

        element_type = \
            ConverterUtil.get_arg(op,
                                  MaceKeyword.mace_element_type_str).i
        if element_type == EltwiseType.SUM.value:
            self.add_min_max_const_node(
                op, op.output[0], True, True, False)
            op.type = HexagonOp.QuantizedAdd_8p8to8.name
        elif element_type == EltwiseType.SUB.value:
            self.add_min_max_const_node(
                op, op.output[0], True, True, False)
            op.type = HexagonOp.QuantizedSub_8p8to8.name
        elif element_type == EltwiseType.PROD.value:
            op.type = HexagonOp.QuantizedMul_8x8to8.name
        else:
            mace_check(False,
                       "Hexagon does not support elementwise %s"
                       % EltwiseType(element_type).name)

    def convert_pooling(self, op):
        self.add_min_max_const_node(op, op.input[0])

        window_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_kernel_str)
        self.add_arg_const_node(
            op, '/window:0', [1, window_arg.ints[0], window_arg.ints[1], 1])
        strides_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_strides_str)
        self.add_arg_const_node(
            op, '/strides:0', [1, strides_arg.ints[0], strides_arg.ints[1], 1])

        pooling_type_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_pooling_type_str)
        if PoolingType(pooling_type_arg.i) == PoolingType.AVG:
            op.type = HexagonOp.QuantizedAvgPool_8.name
        else:
            op.type = HexagonOp.QuantizedMaxPool_8.name

    def convert_quantize(self, op):
        op.type = HexagonOp.QuantizeINPUT_f_to_8.name

    def convert_reduce(self, op):
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
        if len(axis_arg.ints) == 1:
            dim1, dim2 = (input_dims[1], 1) \
                if axis_arg.ints[0] == 1 else (1, input_dims[2])
        else:
            dim1, dim2 = input_dims[1], input_dims[2]
        self.add_arg_const_node(op, '/window:0', [1, dim1, dim2, 1])
        self.add_arg_const_node(op, '/strides:0', [1, dim1, dim2, 1])

        op.type = HexagonOp.QuantizedAvgPool_8.name

    def convert_resizebilinear(self, op):
        newdim_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_resize_size_str)
        self.add_arg_const_node(
            op, '/newdim:0', [len(newdim_arg.ints)], newdim_arg.ints)

        self.add_min_max_const_node(op, op.input[0])

        align_corners_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_align_corners_str)
        self.add_arg_const_node(
            op, '/align_corners:0', [1], [align_corners_arg.i])

        op.type = HexagonOp.QuantizedResizeBilinear_8.name

    def convert_softmax(self, op):
        self.add_min_max_const_node(op, op.input[0])

        op.type = HexagonOp.QuantizedSoftmax_8.name
