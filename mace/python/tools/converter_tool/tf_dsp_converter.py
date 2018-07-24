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
from mace.python.tools import graph_util
from mace.python.tools.convert_util import mace_check

import tensorflow as tf
from tensorflow.core.framework import tensor_shape_pb2
from operator import mul
import numpy as np


class DspOps(object):
    def __init__(self):
        self.dsp_ops = {
            'INPUT': 'INPUT"',
            'OUTPUT': 'OUTPUT',
            'NoOp': 'Nop',
            'FLATTEN': 'Flatten',
            'Identity': 'Nop',
            'Placeholder': 'INPUT',
            'Const': 'Const',
            'QuantizedConv2D': 'QuantizedConv2d_8x8to32',
            'QuantizedMatMul': 'QuantizedMatMul_8x8to32',
            'QuantizeDownAndShrinkRange': 'QuantizeDownAndShrinkRange_32to8',
            'QuantizedRelu': 'QuantizedRelu_8',
            'QuantizedReluX': 'QuantizedReluX_8',
            'QuantizedMaxPool': 'QuantizedMaxPool_8',
            'QuantizedAvgPool': 'QuantizedAvgPool_8',
            'QuantizedConcat': 'QuantizedConcat_8',
            'QuantizedBiasAdd': 'QuantizedBiasAdd_8p8to32',
            'QuantizedResizeBilinear': 'QuantizedResizeBilinear_8',
            'QuantizedSpaceToBatchND': 'QuantizedSpaceToBatchND_8',
            'QuantizedBatchToSpaceND': 'QuantizedBatchToSpaceND_8',
            'QuantizedSoftmax': 'QuantizedSoftmax_8',
            'QuantizedTanh': 'QuantizedTanh_8',
            'Min': 'Min_f',
            'Max': 'Max_f',
            'QuantizeV2': 'Quantize',
            'Dequantize': 'Dequantize',
            'Softmax': 'Softmax_f',
            'Reshape': 'Reshape',
            'QuantizedReshape': 'QuantizedReshape',
            'Sigmoid': 'Sigmoid_f',
            'Slice': 'Slice_f',
            'Add': 'Add_f',
            'Mul': 'Mul_f',
            'Requantize': 'Requantize_32to8',
            'RequantizationRange': 'RequantizationRange_32',
            'Sub': 'Sub_f',
            'Pack': 'Pack_int32',
            'StridedSlice': 'StridedSlice_f',
            'ExpandDims': 'ExpandDims_f',
            'QuantizedMul': 'QuantizedMul_8x8to32',
            'QuantizedAdd': 'QuantizedAdd_8p8to32',
            'Pad': 'Pad_f',
            'SpaceToBatchND': 'SpaceToBatchND_f',
            'BatchToSpaceND': 'BatchToSpaceND_f',
            'ResizeBilinear': 'ResizeBilinear_f',
            'ConcatV2': 'ConcatV2_f',
            'Conv2DBackpropInput': 'Deconv_f',
            'Tanh': 'Tanh_f',
            'Split': 'Split_f',
            'Transpose': 'Transpose_f',
            'Concat': 'Concat_f',
            'AddN': 'AddN_f',
        }

    def has_op(self, tf_op):
        return tf_op in self.dsp_ops

    def map_nn_op(self, tf_op):
        if tf_op not in self.dsp_ops:
            raise Exception('Could not map nn op for: ', tf_op)
        return self.dsp_ops[tf_op]


TF_DTYPE_2_MACE_DTYPE_MAP = {
    tf.float32: mace_pb2.DT_FLOAT,
    tf.half: mace_pb2.DT_HALF,
    tf.int32: mace_pb2.DT_INT32,
    tf.qint32: mace_pb2.DT_INT32,
    tf.quint8: mace_pb2.DT_UINT8,
    tf.uint8: mace_pb2.DT_UINT8,
}


def tf_dtype_2_mace_dtype(tf_dtype):
    mace_dtype = TF_DTYPE_2_MACE_DTYPE_MAP.get(tf_dtype, None)
    if not mace_dtype:
        raise Exception("Not supported tensorflow dtype: " + tf_dtype)
    return mace_dtype


padding_mode = {
    'NA': 0,
    'SAME': 1,
    'VALID': 2,
    'MIRROR_REFLECT': 3,
    'MIRROR_SYMMETRIC': 4,
    'SAME_CAFFE': 5
}


def get_tensor_name_from_op(op_name, port):
    return op_name + ':' + str(port)


def get_node_from_map(op_map, op_or_tensor_name):
    op_name = op_or_tensor_name.split(':')[0]
    return op_map[op_name]


def get_op_and_port_from_tensor(tensor_name):
    op, port = tensor_name.split(':')
    port = int(port)
    return op, port


def max_elem_size(tensor):
    if len(tensor.shape.as_list()) == 0:
        return tensor.dtype.size
    else:
        return reduce(mul, tensor.shape.as_list()) * tensor.dtype.size


def find_dtype(tensor_dtype):
    if tensor_dtype == tf.float32:
        return mace_pb2.DT_FLOAT
    elif tensor_dtype == tf.uint8 or tensor_dtype == tf.quint8:
        return mace_pb2.DT_UINT8
    elif tensor_dtype == tf.int32 or tensor_dtype == tf.qint32:
        return mace_pb2.DT_INT32
    else:
        raise Exception('Unsupported data type: ', tensor_dtype)


def has_padding_and_strides(op):
    return 'padding' in op.node_def.attr and 'strides' in op.node_def.attr


def is_node_flatten_reshape(op):
    return op.type == 'Reshape' and len(op.outputs[0].shape) == 1


def get_input_tensor(op, index):
    input_tensor = op.inputs[index]
    if input_tensor.op.type == 'Reshape':
        input_tensor = get_input_tensor(input_tensor.op, 0)
    return input_tensor


def add_shape_const_node(net_def, op, values, name):
    tensor = net_def.tensors.add()
    node_name = op.name + '/' + name
    tensor.name = node_name + ':0'
    tensor.data_type = mace_pb2.DT_INT32
    tensor.dims.extend(values)
    return tensor.name


def convert_op_outputs(mace_op_def, tf_op):
    mace_op_def.out_max_byte_size.extend(
        [max_elem_size(output) for output in tf_op.outputs])
    mace_op_def.output_type.extend(
        [tf_dtype_2_mace_dtype(output.dtype) for output in tf_op.outputs])
    output_shapes = []
    for output in tf_op.outputs:
        output_shape = mace_pb2.OutputShape()
        shape_list = output.shape.as_list()
        if not shape_list:
            shape_list = [1]
        elif len(shape_list) == 2:
            shape_list = [1, 1, shape_list[0], shape_list[1]]
        output_shape.dims.extend(shape_list)
        output_shapes.append(output_shape)
    mace_op_def.output_shape.extend(output_shapes)


def convert_ops(unresolved_ops, resolved_ops, net_def, dsp_ops):
    first_op = unresolved_ops[0]
    print('Op: ', first_op.name, first_op.type, first_op.outputs[0].shape)

    if first_op.name in resolved_ops:
        pass

    elif first_op.type == 'Const':
        print('Add const node: ', first_op.name)
        tf_tensor = first_op.outputs[0].eval()
        tensor = net_def.tensors.add()
        tensor.name = first_op.outputs[0].name
        tensor.data_type = find_dtype(first_op.outputs[0].dtype)
        shape = list(tf_tensor.shape)
        if len(shape) > 0:
            tensor.dims.extend(shape)
        if first_op.outputs[0].dtype == tf.float32:
            tensor.float_data.extend(tf_tensor.astype(float).flat)
        elif first_op.outputs[0].dtype == tf.int32 or \
                first_op.outputs[0].dtype == tf.int8 or \
                first_op.outputs[0].dtype == tf.int16 or \
                first_op.outputs[0].dtype == tf.quint8 or \
                first_op.outputs[0].dtype == tf.quint16:
            tensor.int32_data.extend(tf_tensor.astype(int).flat)
    elif first_op.type == 'Shape':
        resolved_ops.add(first_op.name)
    else:
        op_def = net_def.op.add()
        op_def.name = first_op.name
        op_def.type = dsp_ops.map_nn_op(first_op.type)
        op_def.padding = padding_mode['NA']

        if len(first_op.outputs) > 0 and first_op.type == 'Dequantize' \
            and len(first_op.outputs[0].consumers()) > 0 \
            and (first_op.outputs[0].consumers()[0].type == 'SpaceToBatchND' or
                         first_op.outputs[0].consumers()[0].type == 'BatchToSpaceND'):  # noqa
            input_tensor = first_op.inputs[0]
            min_tensor = first_op.inputs[1]
            max_tensor = first_op.inputs[2]
            s2b_op = first_op.outputs[0].consumers()[0]
            reshape_op = s2b_op.outputs[0].consumers()[0]
            min_op = reshape_op.outputs[0].consumers()[0]
            max_op = reshape_op.outputs[0].consumers()[1]
            quantize_op = min_op.outputs[0].consumers()[0]
            resolved_ops.add(s2b_op.name)
            resolved_ops.add(reshape_op.name)
            resolved_ops.add(min_op.name)
            resolved_ops.add(max_op.name)
            resolved_ops.add(quantize_op.name)

            op_def.name = quantize_op.name
            op_def.type = dsp_ops.map_nn_op('Quantized' + s2b_op.type)
            op_def.input.append(input_tensor.name)
            op_def.input.extend([t.name for t in s2b_op.inputs[1:]])
            op_def.input.extend([min_tensor.name, max_tensor.name])
            convert_op_outputs(op_def, quantize_op)
        elif (len(first_op.outputs) > 0 and
                first_op.type == 'QuantizedReshape' and
                len(first_op.outputs[0].consumers()) > 0 and
                first_op.outputs[0].consumers()[0].type == 'Dequantize' and
                len(first_op.outputs[0].consumers()[0].outputs[0].consumers()) > 0 and  # noqa
                first_op.outputs[0].consumers()[0].outputs[0].consumers()[0].type == 'Softmax'):  # noqa
            input_tensor = first_op.inputs[0]
            min_tensor = first_op.inputs[2]
            max_tensor = first_op.inputs[3]
            dequantize_op = first_op.outputs[0].consumers()[0]
            softmax_op = dequantize_op.outputs[0].consumers()[0]
            reshape_op = softmax_op.outputs[0].consumers()[0]
            min_op = reshape_op.outputs[0].consumers()[0]
            max_op = reshape_op.outputs[0].consumers()[1]
            quantize_op = min_op.outputs[0].consumers()[0]
            quantize_reshape_op = quantize_op.outputs[0].consumers()[0]

            resolved_ops.add(dequantize_op.name)
            resolved_ops.add(softmax_op.name)
            resolved_ops.add(reshape_op.name)
            resolved_ops.add(min_op.name)
            resolved_ops.add(max_op.name)
            resolved_ops.add(quantize_op.name)
            resolved_ops.add(quantize_reshape_op.name)

            op_def.name = quantize_reshape_op.name
            op_def.type = dsp_ops.map_nn_op('QuantizedSoftmax')
            op_def.input.extend(
                [input_tensor.name, min_tensor.name, max_tensor.name])
            convert_op_outputs(op_def, quantize_reshape_op)
        # remove Squeeze
        elif (len(first_op.outputs) > 0 and
                first_op.type == 'Requantize' and
                len(first_op.outputs[0].consumers()) > 0 and
                first_op.outputs[0].consumers()[0].type == 'Dequantize' and
                len(first_op.outputs[0].consumers()[0].outputs[0].consumers()) > 0 and  # noqa
                first_op.outputs[0].consumers()[0].outputs[0].consumers()[0].type == 'Squeeze'):  # noqa
            dequantize_op = first_op.outputs[0].consumers()[0]
            squeeze_op = dequantize_op.outputs[0].consumers()[0]
            reshape_op = squeeze_op.outputs[0].consumers()[0]
            if reshape_op.type == 'Shape':
                reshape_op = squeeze_op.outputs[0].consumers()[1]
            min_op = reshape_op.outputs[0].consumers()[0]
            max_op = reshape_op.outputs[0].consumers()[1]
            quantize_op = min_op.outputs[0].consumers()[0]

            resolved_ops.add(dequantize_op.name)
            resolved_ops.add(squeeze_op.name)
            resolved_ops.add(reshape_op.name)
            resolved_ops.add(min_op.name)
            resolved_ops.add(max_op.name)
            resolved_ops.add(quantize_op.name)

            op_def.name = quantize_op.name
            op_def.input.extend([t.name for t in first_op.inputs])
            convert_op_outputs(op_def, quantize_op)

            # Squeeze -> Softmax
            next_op = quantize_op.outputs[0].consumers()[0] \
                if len(quantize_op.outputs) > 0 else None
            dequantize_op = next_op.outputs[0].consumers()[0] \
                if next_op and len(next_op.outputs) > 0 and \
                next_op.type == 'QuantizedReshape' and \
                len(next_op.outputs[0].consumers()) > 0 else None
            softmax_op = dequantize_op.outputs[0].consumers()[0] \
                if dequantize_op and len(dequantize_op.outputs) > 0 and \
                dequantize_op.type == 'Dequantize' and \
                len(dequantize_op.outputs[0].consumers()) > 0 else None
            if softmax_op and softmax_op.type == 'Softmax':
                reshape_op = softmax_op.outputs[0].consumers()[0]
                min_op = reshape_op.outputs[0].consumers()[0]
                max_op = reshape_op.outputs[0].consumers()[1]
                quantize_op = min_op.outputs[0].consumers()[0]
                quantize_reshape_op = quantize_op.outputs[0].consumers()[0]

                resolved_ops.add(next_op.name)
                resolved_ops.add(dequantize_op.name)
                resolved_ops.add(softmax_op.name)
                resolved_ops.add(reshape_op.name)
                resolved_ops.add(min_op.name)
                resolved_ops.add(max_op.name)
                resolved_ops.add(quantize_op.name)
                resolved_ops.add(quantize_reshape_op.name)

                softmax_op_def = net_def.op.add()
                softmax_op_def.padding = padding_mode['NA']
                softmax_op_def.name = quantize_reshape_op.name
                softmax_op_def.type = dsp_ops.map_nn_op('QuantizedSoftmax')
                softmax_op_def.input.extend([
                    get_tensor_name_from_op(op_def.name, 0),
                    get_tensor_name_from_op(op_def.name, 1),
                    get_tensor_name_from_op(op_def.name, 2)])
                convert_op_outputs(softmax_op_def, quantize_reshape_op)

        elif len(first_op.outputs) > 0 and first_op.type == 'Dequantize' and \
                len(first_op.outputs[0].consumers()) > 0 and \
                first_op.outputs[0].consumers()[0].type == 'Tanh':
            input_tensor = first_op.inputs[0]
            min_tensor = first_op.inputs[1]
            max_tensor = first_op.inputs[2]
            tanh_op = first_op.outputs[0].consumers()[0]

            # if not last op
            resolved_ops.add(tanh_op.name)
            if tanh_op.outputs[0].consumers():
                reshape_op = tanh_op.outputs[0].consumers()[0]
                min_op = reshape_op.outputs[0].consumers()[0]
                max_op = reshape_op.outputs[0].consumers()[1]
                quantize_op = min_op.outputs[0].consumers()[0]
                resolved_ops.add(reshape_op.name)
                resolved_ops.add(min_op.name)
                resolved_ops.add(max_op.name)
                resolved_ops.add(quantize_op.name)

                op_def.name = quantize_op.name
                op_def.type = dsp_ops.map_nn_op('Quantized' + tanh_op.type)
                op_def.input.extend(
                    [input_tensor.name, min_tensor.name, max_tensor.name])
                convert_op_outputs(op_def, quantize_op)
            # tanh is last op
            else:
                op_def.name = tanh_op.name + '/QuantizedTanh'
                op_def.type = dsp_ops.map_nn_op('Quantized' + tanh_op.type)
                op_def.input.extend(
                    [input_tensor.name, min_tensor.name, max_tensor.name])
                op_def.out_max_byte_size.extend([
                    max_elem_size(input_tensor),
                    max_elem_size(min_tensor),
                    max_elem_size(max_tensor)
                ])
                op_def.output_type.extend(
                    [mace_pb2.DT_UINT8, mace_pb2.DT_FLOAT, mace_pb2.DT_FLOAT])
                output_shapes = []
                for output in first_op.inputs:
                    output_shape = mace_pb2.OutputShape()
                    output_shape.dims.extend(output.shape.as_list())
                    output_shapes.append(output_shape)
                op_def.output_shape.extend(output_shapes)

                new_tanh_op_def = net_def.op.add()
                new_tanh_op_def.name = tanh_op.name
                new_tanh_op_def.type = dsp_ops.map_nn_op('Dequantize')
                new_tanh_op_def.input.extend([
                    get_tensor_name_from_op(op_def.name, 0),
                    get_tensor_name_from_op(op_def.name, 1),
                    get_tensor_name_from_op(op_def.name, 2)
                ])
                convert_op_outputs(new_tanh_op_def, tanh_op)
        elif has_padding_and_strides(first_op):
            op_def.padding = padding_mode[first_op.get_attr('padding')]
            op_def.input.extend([t.name for t in first_op.inputs])
            if 'ksize' in first_op.node_def.attr:
                ksize = first_op.get_attr('ksize')
                ksize_tensor = add_shape_const_node(net_def, first_op, ksize,
                                                    'ksize')
                op_def.input.extend([ksize_tensor])
            strides = first_op.get_attr('strides')
            strides_tensor = add_shape_const_node(net_def, first_op, strides,
                                                  'strides')
            op_def.input.extend([strides_tensor])
            convert_op_outputs(op_def, first_op)
        elif is_node_flatten_reshape(first_op):
            op_def.type = 'Flatten'
            op_def.input.extend([first_op.inputs[0].name])
            convert_op_outputs(op_def, first_op)
        elif dsp_ops.has_op(first_op.type):
            op_def.input.extend([t.name for t in first_op.inputs])
            convert_op_outputs(op_def, first_op)
        else:
            raise Exception('Unsupported op: ', first_op)

        resolved_ops.add(first_op.name)

    del unresolved_ops[0]


def add_output_node(net_def, output_node):
    op_def = net_def.op.add()
    op_def.name = '__output__'
    op_def.type = 'OUTPUT'
    op_def.input.extend([get_tensor_name_from_op(output_node, 0)])


def reverse_batch_to_space_and_biasadd(net_def):
    tensor_map = {}
    for tensor in net_def.tensors:
        tensor_map[tensor.name] = tensor
    op_map = {}
    for op in net_def.op:
        op_map[op.name] = op
    consumers = {}
    for op in net_def.op:
        for ipt in op.input:
            if ipt not in consumers:
                consumers[ipt] = []
            consumers[ipt].append(op)

    new_ops = []
    skip_ops = set()
    visited_ops = set()

    for op in net_def.op:
        if op.name in visited_ops:
            pass
        # pattern: QConv -> RR -> R -> QB2S -> QBiasAdd -> RR -> R
        success = False
        if op.type == 'Requantize_32to8':
            biasadd_requantize_op = op
            biasadd_op = get_node_from_map(op_map,
                                           biasadd_requantize_op.input[0])
            if biasadd_op.type == 'QuantizedBiasAdd_8p8to32':
                b2s_op = get_node_from_map(op_map, biasadd_op.input[0])
                if b2s_op.type == 'QuantizedBatchToSpaceND_8':
                    conv_requantize_op = get_node_from_map(
                        op_map, b2s_op.input[0])
                    conv_op = get_node_from_map(op_map,
                                                conv_requantize_op.input[0])
                    if conv_op.type == 'QuantizedConv2d_8x8to32':
                        new_biasadd_op = mace_pb2.OperatorDef()
                        new_biasadd_op.CopyFrom(biasadd_op)
                        new_biasadd_op.input[0] = get_tensor_name_from_op(
                            conv_requantize_op.name, 0)
                        new_biasadd_op.input[2] = get_tensor_name_from_op(
                            conv_requantize_op.name, 1)
                        new_biasadd_op.input[3] = get_tensor_name_from_op(
                            conv_requantize_op.name, 2)
                        new_biasadd_op.out_max_byte_size[
                            0] = conv_requantize_op.out_max_byte_size[0] * 4

                        new_biasadd_requantize_op = mace_pb2.OperatorDef()
                        new_biasadd_requantize_op.CopyFrom(
                            biasadd_requantize_op)
                        new_biasadd_requantize_op.out_max_byte_size[
                            0] = new_biasadd_op.out_max_byte_size[0] / 4

                        new_b2s_op = mace_pb2.OperatorDef()
                        new_b2s_op.CopyFrom(b2s_op)
                        new_b2s_op.input[0] = get_tensor_name_from_op(
                            biasadd_requantize_op.name, 0)
                        new_b2s_op.input[3] = get_tensor_name_from_op(
                            biasadd_requantize_op.name, 1)
                        new_b2s_op.input[4] = get_tensor_name_from_op(
                            biasadd_requantize_op.name, 2)

                        new_ops.extend([
                            new_biasadd_op, new_biasadd_requantize_op,
                            new_b2s_op
                        ])
                        skip_ops = skip_ops.union([
                            biasadd_op.name, biasadd_requantize_op.name,
                            b2s_op.name
                        ])
                        visited_ops.add(op.name)

                        follow_ops = consumers[get_tensor_name_from_op(
                            biasadd_requantize_op.name, 0)]
                        for follow_op in follow_ops:
                            new_follow_op = mace_pb2.OperatorDef()
                            new_follow_op.CopyFrom(follow_op)
                            for i in xrange(len(follow_op.input)):
                                for k in xrange(3):
                                    if new_follow_op.input[i] == get_tensor_name_from_op(  # noqa
                                        biasadd_requantize_op.name, k):
                                        new_follow_op.input[i] = get_tensor_name_from_op(  # noqa
                                            b2s_op.name, k)
                            new_ops.append(new_follow_op)
                            skip_ops.add(follow_op.name)
                            visited_ops.add(follow_op.name)

        visited_ops.add(op.name)

    new_net_def = mace_pb2.NetDef()
    new_net_def.tensors.extend(tensor_map.values())
    new_net_def.op.extend([op for op in net_def.op if op.name not in skip_ops])
    new_net_def.op.extend(new_ops)

    return new_net_def


def add_node_id(net_def):
    node_id_counter = 0
    node_id_map = {}
    for tensor in net_def.tensors:
        tensor.node_id = node_id_counter
        node_id_counter += 1
        tensor_op, port = get_op_and_port_from_tensor(tensor.name)
        node_id_map[tensor_op] = tensor.node_id

    for op in net_def.op:
        op.node_id = node_id_counter
        node_id_counter += 1
        node_id_map[op.name] = op.node_id
        for ipt in op.input:
            op_name, port = get_op_and_port_from_tensor(ipt)
            node_id = node_id_map[op_name]
            node_input = op.node_input.add()
            node_input.node_id = node_id
            node_input.output_port = int(port)

    return net_def


def add_input_output_info(net_def, input_node, output_node, graph, dtype):
    input_tensor = graph.get_tensor_by_name(
        get_tensor_name_from_op(input_node, 0))
    output_tensor = graph.get_tensor_by_name(
        get_tensor_name_from_op(output_node, 0))

    input_info = net_def.input_info.add()
    input_info.name = input_node
    input_info.dims.extend(input_tensor.shape.as_list())
    input_info.data_type = dtype
    if dtype == mace_pb2.DT_UINT8:
        for i in xrange(2):
            input_info = net_def.input_info.add()
            input_info.dims.extend([1, 1, 1, 1])
            input_info.data_type = mace_pb2.DT_FLOAT

    output_info = net_def.output_info.add()
    output_info.name = output_node
    output_info.dims.extend(output_tensor.shape.as_list())
    output_info.data_type = dtype
    if dtype == mace_pb2.DT_UINT8:
        for i in xrange(2):
            output_info = net_def.output_info.add()
            output_info.dims.extend([1, 1, 1, 1])
            output_info.data_type = mace_pb2.DT_FLOAT

    return net_def


def fuse_quantize(net_def):
    tensor_map = {}
    for tensor in net_def.tensors:
        tensor_map[tensor.name] = tensor
    op_map = {}
    for op in net_def.op:
        op_map[op.name] = op
    consumers = {}
    for op in net_def.op:
        for ipt in op.input:
            if ipt not in consumers:
                consumers[ipt] = []
            consumers[ipt].append(op)

    skip_ops = set()
    new_ops = []
    skip_tensors = set()

    # INPUT->Flatten->Minf, Maxf->Quantize
    for op in net_def.op:
        if op.type == 'INPUT':
            input_op = op
            flatten_op = None
            quantize_op = None
            for o in consumers[get_tensor_name_from_op(input_op.name, 0)]:
                if o.type == 'Flatten':
                    flatten_op = o
                elif o.type == 'Quantize':
                    quantize_op = o
            if quantize_op is not None:
                minf_op, maxf_op = consumers[get_tensor_name_from_op(
                    flatten_op.name, 0)]
                skip_ops = skip_ops.union(
                    [flatten_op.name, minf_op.name, maxf_op.name])
                skip_tensors = skip_tensors.union(
                    [minf_op.input[0], maxf_op.input[0],
                     quantize_op.input[1], quantize_op.input[2]])
                quantize_op.type = 'AutoQuantize'
                del quantize_op.input[1:]

    new_net_def = mace_pb2.NetDef()
    new_net_def.tensors.extend([
        tensor for tensor in net_def.tensors if tensor.name not in skip_tensors
    ])
    new_net_def.op.extend([op for op in net_def.op if op.name not in skip_ops])
    new_net_def.op.extend(new_ops)
    return new_net_def


class TensorflowDspConverter(base_converter.ConverterInterface):
    def __init__(self, option, src_model_file):
        self._option = option
        self._mace_net_def = mace_pb2.NetDef()

        # import tensorflow graph
        tf_graph_def = tf.GraphDef()
        with tf.gfile.Open(src_model_file, 'rb') as f:
            tf_graph_def.ParseFromString(f.read())

        self._placeholders = {}
        self.add_shape_info(tf_graph_def)

        with tf.Session() as session:
            with session.graph.as_default() as graph:
                tf.import_graph_def(tf_graph_def, name='')
                self._tf_graph = graph

    def run(self):
        ops = self._tf_graph.get_operations()
        dsp_ops = DspOps()
        resolved_ops = set()

        mace_check(len(self._option.input_nodes) == 1
                   and len(self._option.output_nodes) == 1,
                   'dsp only support single input and output')
        input_node = self._option.input_nodes.values()[0].name
        output_node = self._option.output_nodes.values()[0].name

        # convert const node
        unresolved_ops = [op for op in ops if op.type == 'Const']
        with tf.Session() as session:
            while len(unresolved_ops) > 0:
                convert_ops(unresolved_ops, resolved_ops, self._mace_net_def,
                            dsp_ops)

            # convert op node
            unresolved_ops = [op for op in ops if op.type != 'Const']
            while len(unresolved_ops) > 0:
                convert_ops(unresolved_ops, resolved_ops, self._mace_net_def,
                            dsp_ops)

            add_output_node(self._mace_net_def, output_node)
            net_def = reverse_batch_to_space_and_biasadd(self._mace_net_def)
            net_def = fuse_quantize(net_def)

            sorted_net_def = graph_util.sort_mace_graph(net_def, '__output__')
            net_def_with_node_id = add_node_id(sorted_net_def)

            dtype = mace_pb2.DT_FLOAT
            final_net_def = add_input_output_info(
                net_def_with_node_id, input_node, output_node,
                self._tf_graph, dtype)

        return final_net_def

    def add_shape_info(self, tf_graph_def):
        for node in tf_graph_def.node:
            for input_node in self._option.input_nodes.values():
                if node.name == input_node.name or \
                            node.name + ':0' == input_node.name:
                    del node.attr['shape'].shape.dim[:]
                    node.attr['shape'].shape.dim.extend([
                        tensor_shape_pb2.TensorShapeProto.Dim(size=i) for i in
                        input_node.shape
                    ])
                    self._placeholders[node.name + ':0'] = \
                        np.zeros(shape=input_node.shape, dtype=float)
