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


from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.converter_tool.base_converter import EltwiseType
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import PaddingMode
from mace.python.tools.converter_tool.base_converter import PoolingType
from mace.python.tools.convert_util import mace_check
from mace.python.tools import graph_util

import copy
from operator import mul


class HexagonOps(object):
    def __init__(self):
        self.hexagon_ops = {
            'Quantize': 'QuantizeINPUT_f_to_8',
            'Dequantize': 'DequantizeOUTPUT_8tof',
            'Concat': 'QuantizedConcat_8',
            'Conv2D': 'Supernode_8x8p32to8',
            'DepthwiseConv2d': 'DepthwiseSupernode_8x8p32to8',
            'ResizeBilinear': 'QuantizedResizeBilinear_8',
            'SpaceToBatchND': 'SpaceToBatchND_8',
            'BatchToSpaceND': 'BatchToSpaceND_8',
            'Softmax': 'QuantizedSoftmax_8',
            'Eltwise': 'Eltwise',
            'Pooling': 'Pooling',
            'Identity': 'Nop',
            'Squeeze': 'Nop',
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
        mace_check(len(self._option.input_nodes) == 1
                   and len(self._option.output_nodes) == 1,
                   'dsp only support single input and output')

        for tensor in self._model.tensors:
            self._consts[tensor.name] = tensor

        # convert op node
        self.convert_ops()

        self.add_input_output_node()

        output_name = MaceKeyword.mace_output_node_name + '_' \
            + self._option.output_nodes.values()[0].name
        output_name = normalize_name(output_name)
        self._model = graph_util.sort_mace_graph(self._model, output_name)

        self.add_node_id()

        return self._model

    def convert_ops(self):
        print("Convert mace graph to hexagon.")
        for op in self._model.op:
            if not self._hexagon_ops.has_op(op.type):
                raise Exception('Unsupported op: ', op)
            print('Op: ', op.name, op.type)
            for i in range(len(op.input)):
                if ':' not in op.input[i]:
                    node_name = op.input[i]
                    op.input[i] += ':0'
                    if node_name in self._quantize_activation_info:
                        self._quantize_activation_info[op.input[i]] = \
                            self._quantize_activation_info[node_name]

            if op.type == MaceOp.Conv2D.name \
                    or op.type == MaceOp.DepthwiseConv2d.name:
                mace_check(len(op.input) == 3,
                           "Missing bias of Conv or Depthwise Conv.")
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
                op.type = 'QuantizedAdd_8p8to8'
            elif op.type == MaceOp.Pooling.name:
                pooling_type_arg = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_pooling_type_str)
                if PoolingType(pooling_type_arg.i) == PoolingType.AVG:
                    op.type = 'QuantizedAvgPool_8'
                else:
                    op.type = 'QuantizedMaxPool_8'
            else:
                op.type = self._hexagon_ops.map_nn_op(op.type)

    def add_min_max(self, name, val):
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
                self.add_min_max(min_tensor_name, minval)
            this_op.input.extend([min_tensor_name])
        if add_max:
            if is_activation and diff_port:
                max_tensor_name = op + ':2'
            else:
                max_tensor_name = op + '_max:0'
                self.add_min_max(max_tensor_name, maxval)
            this_op.input.extend([max_tensor_name])

    def add_shape_const_node(self, op, values, name):
        tensor = self._model.tensors.add()
        node_name = op.name + '/' + name
        tensor.name = node_name + ':0'
        tensor.data_type = mace_pb2.DT_INT32
        tensor.dims.extend(values)
        return tensor.name

    def add_input_output_node(self):
        input_node = self._option.input_nodes.values()[0]
        for op in self._model.op:
            if op.name == input_node.name:
                del op.input[0]
                break

        output_node = None
        if not self._option.check_nodes:
            output_name = self._option.output_nodes.values()[0].name
        else:
            output_name = self._option.check_nodes.values()[0].name
        output_name = normalize_name(output_name)
        for op in self._model.op:
            if op.name.startswith(MaceKeyword.mace_output_node_name) \
                    and op.name.find(output_name) != -1:
                output_node = op
                break
        mace_check(output_node is not None,
                   "mace_output_node_* not found.")
        del output_node.output_shape[:]
        del output_node.output_type[:]
        del output_node.out_max_byte_size[:]

    def add_node_id(self):
        node_id_counter = 0
        node_id_map = {}
        for tensor in self._model.tensors:
            tensor.node_id = node_id_counter
            node_id_counter += 1
            tensor_op, port = get_op_and_port_from_tensor(tensor.name)
            node_id_map[tensor_op] = tensor.node_id

        for op in self._model.op:
            op.node_id = node_id_counter
            node_id_counter += 1
            node_id_map[op.name] = op.node_id
            for ipt in op.input:
                if ipt.startswith(MaceKeyword.mace_input_node_name):
                    ipt = ipt[len(MaceKeyword.mace_input_node_name + '_'):]
                op_name, port = get_op_and_port_from_tensor(ipt)
                node_id = node_id_map[op_name]
                node_input = op.node_input.add()
                node_input.node_id = node_id
                node_input.output_port = int(port)
