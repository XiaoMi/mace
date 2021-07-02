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
from transform.base_converter import ActivationType
from transform.base_converter import ConverterUtil
from transform.base_converter import CoordinateTransformationMode
from transform.base_converter import DeviceType
from transform.base_converter import EltwiseType
from transform.base_converter import FrameworkType
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp
from transform.base_converter import PaddingMode
from transform.base_converter import PadType
from transform.base_converter import PoolingType
from transform.base_converter import ReduceType
from quantize import quantize_util
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
    'QuantizedBatchNorm_8x8p8to8',
    'QuantizedClamp_8',
    'QuantizedConcat_8',
    'QuantizedDiv_8',
    'QuantizedInstanceNorm_8',
    'QuantizedMaxPool_8',
    'QuantizedMaximum_8',
    'QuantizedMean_8',
    'QuantizedMinimum_8',
    'QuantizedMul_8x8to8',
    'QuantizedPad_8',
    'QuantizedRecip_8',
    'QuantizedRelu_8',
    'QuantizedReluX_8',
    'QuantizedReshape',
    'QuantizedResizeBilinear_8',
    'QuantizedSigmoid_8',
    'QuantizedSoftmax_8',
    'QuantizedSplit_8',
    'QuantizedSqrt_8',
    'QuantizedStridedSlice_8',
    'QuantizedSub_8p8to8',
    'QuantizedTanh_8',
    'QuantizedTransposeConv2d_8x8p32to8',
    'QuantizeINPUT_f_to_8',
    'ResizeNearestNeighbor_8',
    'SpaceToBatchND_8',
    'SpaceToDepth_8',
    'SuperFC_8x8p32to8',
    'Supernode_8x8p32to8',
    'Nop',
]

HexagonOp = Enum('HexagonOp', [(op, op) for op in HexagonSupportedOps],
                 type=str)


class HexagonPadding(Enum):
    NN_PAD_NA = 0
    NN_PAD_SAME = 1
    NN_PAD_VALID = 2
    NN_PAD_MIRROR_REFLECT = 3
    NN_PAD_MIRROR_SYMMETRIC = 4
    NN_PAD_SAME_CAFFE = 5


padding_mode = {
    PaddingMode.NA: HexagonPadding.NN_PAD_NA.value,
    PaddingMode.SAME: HexagonPadding.NN_PAD_SAME.value,
    PaddingMode.VALID: HexagonPadding.NN_PAD_VALID.value
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


def add_port_for_tensor(name):
    return name + ':0' if ':' not in name else name


def remove_port_for_tensor(name):
    return name[:-2] if ':0' in name else name


class HexagonConverter(base_converter.ConverterInterface):
    activation_type = {
        ActivationType.RELU.name: HexagonOp.QuantizedRelu_8.name,
        ActivationType.RELUX.name: HexagonOp.QuantizedReluX_8.name,
        ActivationType.TANH.name: HexagonOp.QuantizedTanh_8.name,
        ActivationType.SIGMOID.name: HexagonOp.QuantizedSigmoid_8.name,
    }

    eltwise_type = {
        EltwiseType.SUM.value: HexagonOp.QuantizedAdd_8p8to8.name,
        EltwiseType.SUB.value: HexagonOp.QuantizedSub_8p8to8.name,
        EltwiseType.PROD.value: HexagonOp.QuantizedMul_8x8to8.name,
        EltwiseType.MIN.value: HexagonOp.QuantizedMinimum_8.name,
        EltwiseType.MAX.value: HexagonOp.QuantizedMaximum_8.name,
        EltwiseType.DIV.value: HexagonOp.QuantizedDiv_8.name,
    }

    def __init__(self, option, model, quantize_activation_info):
        self._option = option
        self._model = model
        self._new_ops = []
        self._consts = {}
        self._producers = {}
        self._quantize_activation_info = quantize_activation_info
        self._op_converters = {
            MaceOp.Activation.name: self.convert_activation,
            MaceOp.BatchNorm.name: self.convert_batchnorm,
            MaceOp.BatchToSpaceND.name: self.convert_batchspace,
            MaceOp.Concat.name: self.convert_concat,
            MaceOp.Conv2D.name: self.convert_conv2d,
            MaceOp.Deconv2D.name: self.convert_deconv2d,
            MaceOp.DepthToSpace.name: self.convert_depthspace,
            MaceOp.DepthwiseConv2d.name: self.convert_conv2d,
            MaceOp.Dequantize.name: self.convert_dequantize,
            MaceOp.Eltwise.name: self.convert_elementwise,
            MaceOp.ExpandDims.name: self.convert_expanddims,
            MaceOp.FullyConnected.name: self.convert_fullyconnected,
            MaceOp.InstanceNorm.name: self.convert_instancenorm,
            MaceOp.Pad.name: self.convert_pad,
            MaceOp.Pooling.name: self.convert_pooling,
            MaceOp.Quantize.name: self.convert_quantize,
            MaceOp.Reduce.name: self.convert_reduce,
            MaceOp.ResizeBilinear.name: self.convert_resizebilinear,
            MaceOp.ResizeNearestNeighbor.name:
                self.convert_resizenearestneighbor,
            MaceOp.Softmax.name: self.convert_softmax,
            MaceOp.Split.name: self.convert_split,
            MaceOp.StridedSlice.name: self.convert_stridedslice,
            MaceOp.SpaceToBatchND.name: self.convert_batchspace,
            MaceOp.SpaceToDepth.name: self.convert_depthspace,
        }
        self._framework_type = ConverterUtil.get_arg(
            self._model, MaceKeyword.mace_framework_type_str).i

    def run(self):
        self.add_port_and_construct_producers()

        # convert op node
        self.convert_ops()

        model_inputs = self.convert_input_output_node()

        self.add_node_id(model_inputs)

        self.remove_port()

        return self._model

    def add_port_and_construct_producers(self):
        for tensor in self._model.tensors:
            tensor.name = add_port_for_tensor(tensor.name)
            self._consts[tensor.name] = tensor
        for key in tuple(self._quantize_activation_info):
            name = add_port_for_tensor(key)
            self._quantize_activation_info[name] = \
                self._quantize_activation_info[key]
        for op in self._model.op:
            for i in range(len(op.output)):
                op.output[i] = add_port_for_tensor(op.output[i])
                if op.output[i] not in self._producers:
                    self._producers[op.output[i]] = op

    def get_input_shape(self, tensor_name):
        mace_check(tensor_name in self._producers, "Missing producer.")
        op = self._producers[tensor_name]
        input_shape = None
        for i, output in enumerate(op.output):
            if output == tensor_name:
                input_shape = op.output_shape[i].dims
        mace_check(input_shape is not None, "Missing input shape.")
        return input_shape

    def add_port_for_tensors(self,  tensors):
        for i in range(len(tensors)):
            if ':' not in tensors[i]:
                node_name = tensors[i]
                tensors[i] += ':0'
                if node_name in self._quantize_activation_info:
                    self._quantize_activation_info[tensors[i]] = \
                        self._quantize_activation_info[node_name]

    def remove_port(self):
        if self._framework_type == FrameworkType.TENSORFLOW.value:
            return
        for op in self._model.op:
            for i in range(len(op.input)):
                op.input[i] = remove_port_for_tensor(op.input[i])
            for i in range(len(op.output)):
                op.output[i] = remove_port_for_tensor(op.output[i])
        for tensor in self._model.tensors:
            tensor.name = remove_port_for_tensor(tensor.name)

    def add_quantized_scalar_const_node(self, name, val, op=None):
        if op is not None:
            name = op.name + name
            op.input.append(name)
        if name not in self._consts:
            quantized_tensor = quantize_util.quantize(
                [val], DeviceType.HEXAGON.value, False)
            tensor = self._model.tensors.add()
            self._consts[name] = tensor
            tensor.name = name
            tensor.data_type = mace_pb2.DT_UINT8
            tensor.dims.extend([1])
            tensor.int32_data.extend(quantized_tensor.data)
            tensor.minval = quantized_tensor.minval
            tensor.maxval = quantized_tensor.maxval

    def add_scalar_const_node(self, name, val, op=None):
        if op is not None:
            name = op.name + name
            op.input.append(name)
        if name not in self._consts:
            tensor = self._model.tensors.add()
            self._consts[name] = tensor
            tensor.name = name
            tensor.data_type = mace_pb2.DT_FLOAT
            tensor.dims.extend([1])
            tensor.float_data.extend([val])

    def add_arg_const_node(self, op, name, dims, data=None, insert_index=None,
                           data_type=mace_pb2.DT_INT32):
        arg_tensor = self._model.tensors.add()
        arg_tensor.name = op.name + name
        arg_tensor.data_type = data_type
        arg_tensor.dims.extend(dims)
        if data:
            if data_type == mace_pb2.DT_INT32:
                arg_tensor.int32_data.extend(data)
            else:
                arg_tensor.float_data.extend(data)
        if insert_index is None:
            op.input.append(arg_tensor.name)
        else:
            op.input.insert(insert_index, arg_tensor.name)

    def add_min_max_const_node_for_split(self, this_op, tensor_name):
        op, port = get_op_and_port_from_tensor(tensor_name)
        this_op.input.extend([op + ':2'])
        this_op.input.extend([op + ':3'])

    def add_min_max_const_node(
            self, this_op, tensor_name, add_min=True, add_max=True,
            diff_port=True):
        if tensor_name in self._producers and \
                self._producers[tensor_name].type == \
                HexagonOp.QuantizedSplit_8.name:
            return self.add_min_max_const_node_for_split(this_op, tensor_name)
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
                self.add_scalar_const_node(min_tensor_name, minval)
            this_op.input.extend([min_tensor_name])
        if add_max:
            if is_activation and diff_port:
                max_tensor_name = op + ':2'
            else:
                max_tensor_name = op + '_max:0'
                self.add_scalar_const_node(max_tensor_name, maxval)
            this_op.input.extend([max_tensor_name])

    def add_constant_min_max_for_first_op(self, op):
        minval = self._quantize_activation_info[op.input[0]].minval
        maxval = self._quantize_activation_info[op.input[0]].maxval
        input_op, _ = get_op_and_port_from_tensor(op.input[0])
        input_min = input_op + '_min:0'
        input_max = input_op + '_max:0'
        self.add_scalar_const_node(input_min, minval)
        self.add_scalar_const_node(input_max, maxval)
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

    def add_bias(self, op):
        print('Hexagon conv/deconv/fc requires biasadd, we add it.')
        channels = op.output_shape[0].dims[3]
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
        return bias

    def add_padding_type_for_conv_pooling(self, op, kernels, strides):
        arg = ConverterUtil.get_arg(op, MaceKeyword.mace_padding_str)
        if arg is not None:  # TensorFlow
            op.padding = padding_mode[PaddingMode(arg.i)]
        else:                # PyTorch, Caffe
            input_shape = self.get_input_shape(op.input[0])
            output_shape = op.output_shape[0].dims
            in_h, in_w = input_shape[1], input_shape[2]
            k_h, k_w = kernels[0], kernels[1]
            out_h, out_w = output_shape[1], output_shape[2]

            if (out_h == (in_h - k_h) // strides[0] + 1) and \
                    (out_w == (in_w - k_w) // strides[1] + 1):
                op.padding = HexagonPadding.NN_PAD_VALID.value
            elif (out_h == (in_h - 1) // strides[0] + 1) and \
                    (out_w == (in_w - 1) // strides[1] + 1):
                op.padding = HexagonPadding.NN_PAD_SAME_CAFFE.value
            else:
                mace_check(False,
                           "Hexagon does not support padding type for: %s"
                           % op)

    def convert_ops(self):
        print("Convert mace graph to hexagon.")
        for op in self._model.op:
            mace_check(op.type in self._op_converters,
                       "Mace Hexagon does not support op type %s yet"
                       % op.type)
            self.pre_convert(op)
            post_convert_omitted = self._op_converters[op.type](op)
            if post_convert_omitted is None or not post_convert_omitted:
                self.post_convert(op)

        del self._model.op[:]
        self._model.op.extend(self._new_ops)

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

        if not op.HasField("padding"):
            op.padding = padding_mode[PaddingMode.NA]
            arg = ConverterUtil.get_arg(op, MaceKeyword.mace_padding_str)
            if arg is not None:
                op.padding = padding_mode[PaddingMode(arg.i)]

        self._new_ops.append(op)

    def convert_activation(self, op):
        self.add_min_max_const_node(op, op.input[0])

        act_type = ConverterUtil.get_arg(
            op, MaceKeyword.mace_activation_type_str).s.decode()
        if act_type == ActivationType.RELUX.name:
            x = ConverterUtil.get_arg(
                op, MaceKeyword.mace_activation_max_limit_str).f
            self.add_scalar_const_node("/x:0", x, op)
        try:
            op.type = self.activation_type[act_type]
        except KeyError:
            mace_check(False,
                       "Hexagon does not support activation %s" % act_type)

    def convert_batchnorm(self, op):
        bias = op.input.pop()
        self.add_min_max_const_node(op, op.input[0])
        self.add_min_max_const_node(op, op.input[1])
        op.input.append(bias)
        self.add_min_max_const_node(op, bias)
        self.add_min_max_const_node(
            op, op.output[0], True, True, False)

        op.type = HexagonOp.QuantizedBatchNorm_8x8p8to8.name

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
        if len(op.input) < 3:
            bias = self.add_bias(op)
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

        self.add_padding_type_for_conv_pooling(
            op, self._consts[op.input[1]].dims, strides_arg.ints)

        dilations_arg = ConverterUtil.get_arg(op, 'dilations')
        mace_check(dilations_arg is None or
                   (dilations_arg.ints[0] == 1 and dilations_arg.ints[1] == 1),
                   "Hexagon only support dilations[1,1].")

        if op.type == MaceOp.DepthwiseConv2d.name:
            op.type = HexagonOp.DepthwiseSupernode_8x8p32to8.name
        else:
            op.type = HexagonOp.Supernode_8x8p32to8.name

    def add_deconv_pad_node(self, op):
        padding_type_arg = \
            ConverterUtil.get_arg(op, MaceKeyword.mace_padding_type_str)
        padding_values_arg = \
            ConverterUtil.get_arg(op, MaceKeyword.mace_padding_values_str)
        mace_check(padding_type_arg is not None or
                   padding_values_arg is not None,
                   "Missing padding of Deconv.")
        if padding_type_arg is not None:
            padding_type = PaddingMode(padding_type_arg.i)
            strides_arg = ConverterUtil.get_arg(op,
                                                MaceKeyword.mace_strides_str)
            mace_check(strides_arg is not None, "Missing strides of Deconv.")
            stride_h = strides_arg.ints[0]
            stride_w = strides_arg.ints[1]

            input_shape = self.get_input_shape(op.input[0])
            input_h = input_shape[1]
            input_w = input_shape[2]
            filter_tensor = self._consts[op.input[1]]
            filter_h = filter_tensor.dims[1]
            filter_w = filter_tensor.dims[2]
            output_h = op.output_shape[0].dims[1]
            output_w = op.output_shape[0].dims[2]

            if padding_type == PaddingMode.VALID:
                expected_input_h = (output_h - filter_h + stride_h) // stride_h
                expected_input_w = (output_w - filter_w + stride_w) // stride_w
            elif padding_type == PaddingMode.SAME:
                expected_input_h = (output_h + stride_h - 1) // stride_h
                expected_input_w = (output_w + stride_w - 1) // stride_w
            else:
                raise Exception(
                    'Hexagon deconv does not support padding type: ',
                    padding_type)
            mace_check(expected_input_h == input_h,
                       "Wrong input/output height")
            mace_check(expected_input_w == input_w, "Wrong input/output width")

            pad_h = (input_h - 1) * stride_h + filter_h - output_h
            pad_w = (input_w - 1) * stride_w + filter_w - output_w
        else:
            pad_h = padding_values_arg.ints[0]
            pad_w = padding_values_arg.ints[1]

        pad_h, pad_w = max(pad_h, 0), max(pad_w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        paddings = [pad_top, pad_bottom, pad_left, pad_right]
        self.add_arg_const_node(op, "/paddings:0", [1, 1, 2, 2], paddings)

    def convert_deconv2d(self, op):
        if self._framework_type == FrameworkType.TENSORFLOW.value:
            if len(op.input) < 4:
                bias = self.add_bias(op)
            else:
                bias = op.input.pop()
            op.input.pop()  # output shape
        else:
            if len(op.input) < 3:
                bias = self.add_bias(op)
            else:
                bias = op.input.pop()

        self.add_min_max_const_node(op, op.input[0])
        self.add_min_max_const_node(op, op.input[1])

        self.add_deconv_pad_node(op)

        strides_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_strides_str)
        mace_check(strides_arg is not None, "Missing strides of Deconv.")
        self.add_arg_const_node(
            op, '/strides:0', [1, strides_arg.ints[0], strides_arg.ints[1], 1])

        op.input.append(bias)
        self.add_min_max_const_node(op, bias)
        self.add_min_max_const_node(
            op, op.output[0], True, True, False)

        op.type = HexagonOp.QuantizedTransposeConv2d_8x8p32to8.name

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
        element_type = ConverterUtil.get_arg(
            op, MaceKeyword.mace_element_type_str).i

        if element_type == EltwiseType.DIV.value and \
                op.input[0] in self._consts:
            tensor = self._consts[op.input[0]]
            if len(tensor.int32_data) == 1:
                f = tensor.scale * (tensor.int32_data[0] - tensor.zero_point)
                if abs(f - 1) < 1e-6:  # recip
                    op_input = op.input[1]
                    del op.input[:]
                    op.input.append(op_input)
                    self.add_min_max_const_node(op, op.input[0])
                    op.type = HexagonOp.QuantizedRecip_8.name
                    return
        if element_type == EltwiseType.POW.value and \
                ConverterUtil.get_arg(
                    op, MaceKeyword.mace_scalar_input_str).f == 0.5:
            self.add_min_max_const_node(op, op.input[0])
            op.type = HexagonOp.QuantizedSqrt_8.name
            return
        if element_type == EltwiseType.CLIP.value:
            self.add_min_max_const_node(op, op.input[0])
            coeff = ConverterUtil.get_arg(
                    op, MaceKeyword.mace_coeff_str).floats
            min_value, max_value = coeff[0], coeff[1]
            self.add_arg_const_node(op, "/min:0", [1], [min_value],
                                    data_type=mace_pb2.DT_FLOAT)
            self.add_arg_const_node(op, "/max:0", [1], [max_value],
                                    data_type=mace_pb2.DT_FLOAT)
            op.type = HexagonOp.QuantizedClamp_8.name
            return
        if len(op.input) == 1:
            scalar_input = ConverterUtil.get_arg(
                op, MaceKeyword.mace_scalar_input_str).f
            self.add_quantized_scalar_const_node("/b:0", scalar_input, op)
        self.add_min_max_const_node(op, op.input[0])
        self.add_min_max_const_node(op, op.input[1])

        if element_type in [EltwiseType.SUM.value,
                            EltwiseType.SUB.value,
                            EltwiseType.MIN.value,
                            EltwiseType.MAX.value,
                            EltwiseType.DIV.value]:
            self.add_min_max_const_node(
                op, op.output[0], True, True, False)
        try:
            op.type = self.eltwise_type[element_type]
        except KeyError:
            mace_check(False,
                       "Hexagon does not support elementwise %s"
                       % EltwiseType(element_type).name)

    def convert_expanddims(self, op):
        shape = op.output_shape[0].dims
        self.add_arg_const_node(op, '/shape:0', [len(shape)], shape)

        self.add_min_max_const_node(op, op.input[0])
        # Convert to reshape as hexagon does not support quantized expanddims
        op.type = HexagonOp.QuantizedReshape.name

    def convert_fullyconnected(self, op):
        if len(op.input) < 3:
            bias = self.add_bias(op)
        else:
            bias = op.input.pop()
        self.add_min_max_const_node(op, op.input[0])
        self.add_min_max_const_node(op, op.input[1])
        op.input.append(bias)
        self.add_min_max_const_node(op, bias)
        self.add_min_max_const_node(
            op, op.output[0], True, True, False)

        op.type = HexagonOp.SuperFC_8x8p32to8.name

    def convert_instancenorm(self, op):
        affine = ConverterUtil.get_arg(op, MaceKeyword.mace_affine_str).i
        if not affine:
            del op.input[1:]
            self.add_min_max_const_node(op, op.input[0])
            op.type = HexagonOp.QuantizedInstanceNorm_8.name
        else:
            mace_check(False,
                       "Hexagon does not support instancenorm with affine")

    def convert_pad(self, op):
        self.add_min_max_const_node(op, op.input[0])

        paddings = ConverterUtil.get_arg(
            op, MaceKeyword.mace_paddings_str).ints
        self.add_arg_const_node(
            op, '/paddings:0', [1, 1, len(paddings) // 2, 2], paddings)

        pad_type_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_pad_type_str)
        mace_check(pad_type_arg is None or
                   pad_type_arg.i == PadType.CONSTANT.value,
                   "Hexagon only supports constant pad")
        constant_value = ConverterUtil.get_arg(
            op, MaceKeyword.mace_constant_value_str).f
        self.add_scalar_const_node('/constant_value:0', constant_value, op)

        op.type = HexagonOp.QuantizedPad_8.name

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

        self.add_padding_type_for_conv_pooling(
            op, window_arg.ints, strides_arg.ints)

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

        self.add_arg_const_node(op, '/axes:0', [len(axis_arg.ints)],
                                axis_arg.ints)
        self.add_min_max_const_node(op, op.output[0], True, True, False)

        op.type = HexagonOp.QuantizedMean_8.name

    def add_resize_args(self, op):
        align_corners_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_align_corners_str)

        if align_corners_arg:
            self.add_arg_const_node(
                op, '/align_corners:0', [1], [align_corners_arg.i])
        else:
            self.add_arg_const_node(
                op, '/align_corners:0', [1], [0])

        coordinate_transformation_mode_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_coordinate_transformation_mode_str)
        if coordinate_transformation_mode_arg is not None:
            name = CoordinateTransformationMode(
                coordinate_transformation_mode_arg.i)
            value = coordinate_transformation_mode_arg.i
            if (value == CoordinateTransformationMode.HALF_PIXEL.value
                or value == CoordinateTransformationMode.PYTORCH_HALF_PIXEL.value):  # noqa
                self.add_arg_const_node(
                    op, '/half_pixel_centers:0', [1], [1])
            else:
                mace_check(False, "Unsupported coordinate_transformation_mode")

    def convert_resizebilinear(self, op):
        resize_size_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_resize_size_str)
        if resize_size_arg is not None:
            newdim = resize_size_arg.ints
        else:
            height_scale_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_height_scale_str)
            width_scale_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_width_scale_str)
            mace_check(height_scale_arg is not None and
                       width_scale_arg is not None,
                       "Wrong ResizeBilinear arguments.")
            if len(op.input) == 2:
                op.input.pop()
            height_scale = height_scale_arg.f
            width_scale = width_scale_arg.f
            producer_op = self._producers[op.input[0]]
            for i in range(len(producer_op.output)):
                if producer_op.output[i] == op.input[0]:
                    input_shape = producer_op.output_shape[i]
                    break
            newdim = [int(height_scale * input_shape.dims[1]),
                      int(width_scale * input_shape.dims[2])]
        self.add_arg_const_node(op, '/newdim:0', [2], newdim)

        self.add_min_max_const_node(op, op.input[0])

        self.add_resize_args(op)

        op.type = HexagonOp.QuantizedResizeBilinear_8.name

    def convert_resizenearestneighbor(self, op):
        height_scale_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_height_scale_str)
        width_scale_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_width_scale_str)
        if height_scale_arg is not None:
            mace_check(width_scale_arg is not None,
                       "height scale and width scale should be present at the same time.")  # noqa
            if len(op.input) == 2:
                op.input.pop()
            height_scale = height_scale_arg.f
            width_scale = width_scale_arg.f
            producer_op = self._producers[op.input[0]]
            for i in range(len(producer_op.output)):
                if producer_op.output[i] == op.input[0]:
                    input_shape = producer_op.output_shape[i]
                    break
            newdim = [int(height_scale * input_shape.dims[1]),
                      int(width_scale * input_shape.dims[2])]
            self.add_arg_const_node(op, '/newdim:0', [2], newdim)

        self.add_min_max_const_node(op, op.input[0])

        self.add_resize_args(op)

        op.type = HexagonOp.ResizeNearestNeighbor_8.name

    def convert_softmax(self, op):
        self.add_min_max_const_node(op, op.input[0])

        op.type = HexagonOp.QuantizedSoftmax_8.name

    def convert_split(self, op):
        op_input = op.input[0]
        del op.input[:]
        axis_value = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str).i
        self.add_arg_const_node(op, '/axis:0', [1], [axis_value])
        op.input.append(op_input)
        self.add_min_max_const_node(op, op_input)

        for i in range(len(op.output) - 1):
            op.output_type.append(mace_pb2.DT_UINT8)

        op.type = HexagonOp.QuantizedSplit_8.name

    def convert_stridedslice(self, op):
        begin_mask = ConverterUtil.get_arg(
            op, MaceKeyword.mace_begin_mask_str).i
        end_mask = ConverterUtil.get_arg(
            op, MaceKeyword.mace_end_mask_str).i
        shrink_mask = ConverterUtil.get_arg(
            op, MaceKeyword.mace_shrink_axis_mask_str).i
        self.add_arg_const_node(op, "/begin_mask:0", [1], [begin_mask])
        self.add_arg_const_node(op, "/end_mask:0", [1], [end_mask])
        self.add_arg_const_node(op, "/shrink_mask:0", [1], [shrink_mask])
        self.add_min_max_const_node(op, op.input[0])

        op.type = HexagonOp.QuantizedStridedSlice_8.name
