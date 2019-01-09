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


import sys
from enum import Enum
import six

from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter
from mace.python.tools.converter_tool.base_converter import PoolingType
from mace.python.tools.converter_tool.base_converter import PaddingMode
from mace.python.tools.converter_tool.base_converter import ActivationType
from mace.python.tools.converter_tool.base_converter import EltwiseType
from mace.python.tools.converter_tool.base_converter import ReduceType
from mace.python.tools.converter_tool.base_converter import FrameworkType
from mace.python.tools.converter_tool.base_converter import RoundMode
from mace.python.tools.converter_tool.base_converter import DataFormat
from mace.python.tools.converter_tool.base_converter import FilterFormat
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.convert_util import mace_check

import onnx
import onnx.utils
from onnx import helper, shape_inference, numpy_helper, optimizer
import numpy as np
from onnx import mapping
from onnx import TensorProto
from numbers import Number


OnnxSupportedOps = [
    'Abs',
    # 'Acos',
    # 'Acosh',
    'Add',
    # 'And',
    'ArgMax',
    'ArgMin',
    # 'Asin',
    # 'Asinh',
    # 'Atan',
    # 'Atanh',
    'AveragePool',
    'BatchNormalization',
    'Cast',
    # 'Ceil',
    # 'Clip',
    # 'Compress',
    'Concat',
    # 'Constant',
    # 'ConstantLike',
    'Conv',
    'ConvTranspose',
    # 'Cos',
    # 'Cosh',
    'DepthToSpace',
    'Div',
    'Dropout',
    'Elu',
    'Equal',
    # 'Exp',
    # 'Expand',
    # 'EyeLike',
    # 'Flatten',
    # 'Floor',
    # 'GRU',
    'Gather',
    'Gemm',
    'GlobalAveragePool',
    # 'GlobalLpPool',
    'GlobalMaxPool',
    # 'Greater',
    # 'HardSigmoid',
    # 'Hardmax',
    'Identity',
    # 'If',
    'ImageScaler',
    # 'InstanceNormalization',
    # 'LRN',
    # 'LSTM',
    'LeakyRelu',
    # 'Less',
    # 'Log',
    # 'LogSoftmax',
    # 'Loop',
    # 'LpNormalization',
    # 'LpPool',
    'MatMul',
    'Max',
    'MaxPool',
    # 'MaxRoiPool',
    # 'MaxUnpool',
    'Mean',
    'Min',
    'Mul',
    # 'Multinomial',
    'Neg',
    # 'Not',
    # 'OneHot',
    # 'Or',
    'PRelu',
    'Pad',
    'Pow',
    # 'RNN',
    # 'RandomNormal',
    # 'RandonNormalLike',
    # 'RandonUniform',
    # 'RandonUniformLike',
    'Reciprocal',
    # 'ReduceL1',
    # 'ReduceL2',
    # 'ReduceLogSum',
    # 'ReduceLogSumExp',
    'ReduceMax',
    'ReduceMean',
    'ReduceMin',
    'ReduceProd',
    # 'ReduceSum',
    # 'ReduceSumSquare',
    'Relu',
    'Reshape',
    # 'Scan',
    # 'Selu',
    'Shape',
    'Sigmoid',
    # 'Sin',
    # 'Sinh',
    # 'Size',
    # 'Slice',
    'Softmax',
    # 'Softplus',
    # 'Softsign',
    'SpaceToDepth',
    'Split',
    'Sqrt',
    'Squeeze',
    'Sub',
    'Sum',
    # 'Tan',
    'Tanh',
    # 'Tile',
    # 'TopK',
    'Transpose',
    # 'Unsqueeze',
    # 'Upsample',
    # 'Xor',
]

OnnxOpType = Enum('OnnxOpType',
                  [(op, op) for op in OnnxSupportedOps],
                  type=str)

onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: data_type.onnx2tf(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: data_type.onnx2tf(x),
}


def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)


def convert_onnx(attr):
    return convert_onnx_attribute_proto(attr)


def convert_onnx_attribute_proto(attr_proto):
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return str(attr_proto.s, 'utf-8')\
            if sys.version_info.major == 3 else attr_proto.s
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        if IS_PYTHON3:
            str_list = map(lambda x: str(x, 'utf-8'), str_list)
        return str_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))


def onnx_dtype(dtype):
    if isinstance(dtype, Number):
        onnx_dtype = dtype
    elif isinstance(dtype, str):
        onnx_dtype = TensorProto.DataType.Value(dtype)
    else:
        raise RuntimeError("dtype should be number or str.")
    return mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]


class OnnxNode(object):
    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.domain = str(node.domain)
        self.attrs = dict([(attr.name,
                            translate_onnx(attr.name, convert_onnx(attr)))
                           for attr in node.attribute])
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.node_proto = node

    def print_info(self):
        print "node: ", self.name
        print "    type: ", self.op_type
        print "    domain: ", self.domain
        print "    inputs: ", self.inputs
        print "    outputs: ", self.outputs
        print "    attrs:"
        for arg in self.attrs:
            print "        %s: %s" % (arg, self.attrs[arg])


class OnnxTensor(object):
    def __init__(self, name, value, shape, dtype):
        self._name = name
        self._tensor_data = value
        self._shape = shape
        self._dtype = dtype


class OnnxConverter(base_converter.ConverterInterface):
    pooling_type_mode = {
        OnnxOpType.AveragePool.name: PoolingType.AVG,
        OnnxOpType.MaxPool.name: PoolingType.MAX
    }

    auto_pad_mode = {
        'NOTSET': PaddingMode.NA,
        'SAME_UPPER': PaddingMode.SAME,
        'SAME_LOWER': PaddingMode.SAME,
        'VALID': PaddingMode.VALID,
    }
    auto_pad_mode = {six.b(k): v for k, v in six.iteritems(auto_pad_mode)}

    eltwise_type = {
        OnnxOpType.Mul.name: EltwiseType.PROD,
        OnnxOpType.Add.name: EltwiseType.SUM,
        OnnxOpType.Max.name: EltwiseType.MAX,
        OnnxOpType.Min.name: EltwiseType.MIN,
        OnnxOpType.Abs.name: EltwiseType.ABS,
        OnnxOpType.Pow.name: EltwiseType.POW,
        OnnxOpType.Sub.name: EltwiseType.SUB,
        OnnxOpType.Div.name: EltwiseType.DIV,
        OnnxOpType.Neg.name: EltwiseType.NEG,
        OnnxOpType.Sum.name: EltwiseType.SUM,
        OnnxOpType.Equal.name: EltwiseType.EQUAL,
        OnnxOpType.Sqrt.name: EltwiseType.POW,
        OnnxOpType.Reciprocal.name: EltwiseType.POW,
    }

    reduce_type = {
        OnnxOpType.GlobalAveragePool.name: ReduceType.MEAN,
        OnnxOpType.GlobalMaxPool.name: ReduceType.MAX,
        OnnxOpType.ReduceMax.name: ReduceType.MAX,
        OnnxOpType.ReduceMean.name: ReduceType.MEAN,
        OnnxOpType.ReduceMin.name: ReduceType.MIN,
        OnnxOpType.ReduceProd.name: ReduceType.PROD,
    }

    activation_type = {
        OnnxOpType.Relu.name: ActivationType.RELU,
        OnnxOpType.LeakyRelu.name: ActivationType.LEAKYRELU,
        OnnxOpType.PRelu.name: ActivationType.PRELU,
        OnnxOpType.Tanh.name: ActivationType.TANH,
        OnnxOpType.Sigmoid.name: ActivationType.SIGMOID,
    }

    def __init__(self, option, src_model_file):
        self._op_converters = {
            OnnxOpType.Abs.name: self.convert_eltwise,
            OnnxOpType.Add.name: self.convert_eltwise,
            OnnxOpType.ArgMax.name: self.convert_argmax,
            OnnxOpType.ArgMin.name: self.convert_argmax,
            OnnxOpType.AveragePool.name: self.convert_pooling,
            OnnxOpType.BatchNormalization.name: self.convert_fused_batchnorm,
            OnnxOpType.Cast.name: self.convert_cast,
            OnnxOpType.Concat.name: self.convert_concat,
            OnnxOpType.Conv.name: self.convert_conv2d,
            OnnxOpType.ConvTranspose.name: self.convert_deconv,
            OnnxOpType.DepthToSpace.name: self.convert_depth_space,
            OnnxOpType.Dropout.name: self.convert_identity,
            OnnxOpType.Div.name: self.convert_eltwise,
            OnnxOpType.Equal.name: self.convert_eltwise,
            OnnxOpType.Gather.name: self.convert_gather,
            OnnxOpType.Gemm.name: self.convert_gemm,
            OnnxOpType.GlobalAveragePool.name: self.convert_reduce,
            OnnxOpType.GlobalMaxPool.name: self.convert_reduce,
            OnnxOpType.Identity.name: self.convert_identity,
            OnnxOpType.ImageScaler.name: self.convert_imagescaler,
            OnnxOpType.LeakyRelu.name: self.convert_activation,
            OnnxOpType.Max.name: self.convert_eltwise,
            OnnxOpType.MaxPool.name: self.convert_pooling,
            OnnxOpType.MatMul.name: self.convert_matmul,
            OnnxOpType.Min.name: self.convert_eltwise,
            OnnxOpType.Mul.name: self.convert_eltwise,
            OnnxOpType.Neg.name: self.convert_eltwise,
            OnnxOpType.Pad.name: self.convert_pad,
            OnnxOpType.Pow.name: self.convert_eltwise,
            OnnxOpType.PRelu.name: self.convert_activation,
            OnnxOpType.Relu.name: self.convert_activation,
            OnnxOpType.Reshape.name: self.convert_reshape,
            OnnxOpType.Reciprocal.name: self.convert_eltwise,
            OnnxOpType.Sigmoid.name: self.convert_activation,
            OnnxOpType.Softmax.name: self.convert_softmax,
            OnnxOpType.SpaceToDepth.name: self.convert_depth_space,
            OnnxOpType.Split.name: self.convert_split,
            OnnxOpType.Sqrt.name: self.convert_eltwise,
            OnnxOpType.Squeeze.name: self.convert_squeeze,
            OnnxOpType.Sub.name: self.convert_eltwise,
            OnnxOpType.Sum.name: self.convert_eltwise,
            OnnxOpType.Tanh.name: self.convert_activation,
            OnnxOpType.Transpose.name: self.convert_transpose,
        }
        self._option = option
        self._mace_net_def = mace_pb2.NetDef()
        ConverterUtil.set_filter_format(self._mace_net_def, FilterFormat.OIHW)
        onnx_model = onnx.load(src_model_file)

        polished_model = onnx.utils.polish_model(onnx_model)

        print "onnx model IR version: ", onnx_model.ir_version
        print "onnx model opset import: ", onnx_model.opset_import

        self._onnx_model = shape_inference.infer_shapes(polished_model)
        self._graph_shapes_dict = {}
        self._consts = {}
        self._replace_tensors = {}

    def print_graph_info(self, graph):
        for value_info in graph.value_info:
            print "value info:", value_info
        for value_info in graph.input:
            print "inputs info:", value_info
        for value_info in graph.output:
            print "outputs info:", value_info

    def extract_shape_info(self, graph):
        def extract_value_info(shape_dict, value_info):
            t = tuple([int(dim.dim_value)
                       for dim in value_info.type.tensor_type.shape.dim])
            if t:
                shape_dict[value_info.name] = t

        for value_info in graph.value_info:
            extract_value_info(self._graph_shapes_dict, value_info)
        for value_info in graph.input:
            extract_value_info(self._graph_shapes_dict, value_info)
        for value_info in graph.output:
            extract_value_info(self._graph_shapes_dict, value_info)

    def add_tensor(self, name, shape, data_type, value):
        tensor = self._mace_net_def.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_type = data_type
        tensor.float_data.extend(value.flat)

    def run(self):
        graph_def = self._onnx_model.graph
        self.extract_shape_info(graph_def)
        self.convert_tensors(graph_def)
        self.convert_ops(graph_def)
        # self.print_graph_info(graph_def)
        # shape_inferer = mace_shape_inference.ShapeInference(
        #     self._mace_net_def,
        #     self._option.input_nodes.values())
        # shape_inferer.run()
        return self._mace_net_def

    def add_stride_pad_kernel_arg(self, attrs, op_def):
        if 'strides' in attrs:
            strides = attrs['strides']
            mace_check(len(strides) == 2, "strides should has 2 values.")
            stride = [strides[0], strides[1]]
        else:
            stride = [1, 1]

        strides_arg = op_def.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(stride)

        if 'kernel_shape' in attrs:
            kernel_shape = attrs['kernel_shape']
            mace_check(len(kernel_shape) == 2,
                       "kernel shape should has 2 values.")
            kernel = [kernel_shape[0], kernel_shape[1]]
            kernels_arg = op_def.arg.add()
            kernels_arg.name = MaceKeyword.mace_kernel_str
            kernels_arg.ints.extend(kernel)

        # TODO: Does not support AutoPad yet.
        if 'pads' in attrs:
            pads = attrs['pads']
            if len(pads) == 4:
                pad = [pads[0] + pads[2], pads[1] + pads[3]]
            else:
                pad = [0, 0]
            padding_arg = op_def.arg.add()
            padding_arg.name = MaceKeyword.mace_padding_values_str
            padding_arg.ints.extend(pad)
        elif 'auto_pad' in attrs:
            auto_pad_arg = op_def.arg.add()
            auto_pad_arg.name = MaceKeyword.mace_padding_str
            auto_pad_arg.i = self.auto_pad_mode[attrs['auto_pad']].value
        else:
            pad = [0, 0]
            padding_arg = op_def.arg.add()
            padding_arg.name = MaceKeyword.mace_padding_values_str
            padding_arg.ints.extend(pad)

    def convert_ops(self, graph_def):
        for n in graph_def.node:
            node = OnnxNode(n)
            mace_check(node.op_type in self._op_converters,
                       "Mace does not support onnx op type %s yet"
                       % node.op_type)
            self._op_converters[node.op_type](node)

    def convert_tensors(self, graph_def):
        initializer = graph_def.initializer
        if initializer:
            for init in initializer:
                tensor = self._mace_net_def.tensors.add()
                tensor.name = init.name

                onnx_tensor = numpy_helper.to_array(init)
                tensor.dims.extend(list(init.dims))
                data_type = onnx_dtype(init.data_type)

                if data_type == np.float32 or data_type == np.float64:
                    tensor.data_type = mace_pb2.DT_FLOAT
                    tensor.float_data.extend(
                        onnx_tensor.astype(np.float32).flat)
                elif data_type == np.int32:
                    tensor.data_type = mace_pb2.DT_INT32
                    tensor.int32_data.extend(
                        onnx_tensor.astype(np.int32).flat)
                elif data_type == np.int64:
                    tensor.data_type = mace_pb2.DT_INT32
                    tensor.int32_data.extend(
                        onnx_tensor.astype(np.int32).flat)
                else:
                    mace_check(False,
                               "Not supported tensor type: %s" % data_type)
                self._consts[tensor.name] = tensor

    def convert_general_op(self, node):
        op = self._mace_net_def.op.add()
        op.name = node.name

        for input in node.inputs:
            if input in self._replace_tensors:
                input = self._replace_tensors[input]
            op.input.append(input)
        for output in node.outputs:
            op.output.append(output)
            output_shape = op.output_shape.add()
            shape_info = self._graph_shapes_dict[output]
            output_shape.dims.extend(shape_info)

        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        data_type_arg.i = self._option.data_type

        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.ONNX.value

        ConverterUtil.add_data_format_arg(op, DataFormat.NCHW)
        return op

    def convert_fused_batchnorm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.BatchNorm.name

        if "epsilon" in node.attrs:
            epsilon_value = node.attrs["epsilon"]
        else:
            epsilon_value = 1e-5

        mace_check(len(node.inputs) == 5, "batch norm should have 5 inputs.")

        gamma_value = np.array(self._consts[node.inputs[1]].float_data)
        beta_value = np.array(self._consts[node.inputs[2]].float_data)
        mean_value = np.array(self._consts[node.inputs[3]].float_data)
        var_value = np.array(self._consts[node.inputs[4]].float_data)

        scale_name = node.name + 'scale'
        offset_name = node.name + 'offset'
        scale_value = (
                (1.0 / np.sqrt(
                    var_value + epsilon_value)) * gamma_value)
        offset_value = (-mean_value * scale_value) + beta_value
        self.add_tensor(scale_name, scale_value.shape, mace_pb2.DT_FLOAT,
                        scale_value)
        self.add_tensor(offset_name, offset_value.shape, mace_pb2.DT_FLOAT,
                        offset_value)
        del op.input[1:]
        op.input.extend([scale_name, offset_name])
        del op.output[1:]
        del op.output_shape[1:]

    def convert_conv2d(self, node):
        op = self.convert_general_op(node)
        self.add_stride_pad_kernel_arg(node.attrs, op)
        group_arg = op.arg.add()
        group_arg.name = MaceKeyword.mace_group_str
        if 'group' in node.attrs:
            group_val = node.attrs["group"]
        else:
            group_val = 1
        group_arg.i = group_val

        is_depthwise = False
        if group_val > 1:
            filter_shape = self._graph_shapes_dict[node.inputs[1]]
            mace_check(group_val == filter_shape[0] and
                       filter_shape[1] == 1,
                       "Mace does not support group convolution yet")
            filter_tensor = self._consts[node.inputs[1]]
            new_shape = [filter_shape[1], filter_shape[0],
                         filter_shape[2], filter_shape[3]]
            del filter_tensor.dims[:]
            filter_tensor.dims.extend(new_shape)
            is_depthwise = True
        if is_depthwise:
            op.type = MaceOp.DepthwiseConv2d.name
        else:
            op.type = MaceOp.Conv2D.name

        dilation_arg = op.arg.add()
        dilation_arg.name = MaceKeyword.mace_dilations_str
        if 'dilations' in node.attrs:
            dilation_val = node.attrs["dilations"]
        else:
            dilation_val = [1, 1]
        dilation_arg.ints.extend(dilation_val)

    def convert_biasadd(self, node):
        self.convert_general_op(node)

    def convert_concat(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Concat.name
        mace_check('axis' in node.attrs,
                   'Concat op should have axis attribute.')
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = node.attrs['axis']
        axis_arg.i = 4 + axis_arg.i if axis_arg.i < 0 else axis_arg.i
        mace_check(axis_arg.i == 1,
                   "only support concat at channel dimension")

    def convert_activation(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Activation.name

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_activation_type_str
        type_arg.s = six.b(self.activation_type[node.op_type].name)

        if "alpha" in node.attrs:
            alpha_value = node.attrs["alpha"]
        else:
            if node.op_type == OnnxOpType.LeakyRelu.name:
                alpha_value = 0.01
            else:
                alpha_value = 0
        alpha_arg = op.arg.add()
        alpha_arg.name = MaceKeyword.mace_activation_max_limit_str
        alpha_arg.f = alpha_value

    def convert_pooling(self, node):
        op = self.convert_general_op(node)

        op.type = MaceOp.Pooling.name
        self.add_stride_pad_kernel_arg(node.attrs, op)
        pooling_type_arg = op.arg.add()
        pooling_type_arg.name = MaceKeyword.mace_pooling_type_str
        pooling_type_arg.i = self.pooling_type_mode[node.op_type].value

        round_mode_arg = op.arg.add()
        round_mode_arg.name = MaceKeyword.mace_round_mode_str
        round_mode_arg.i = RoundMode.FLOOR.value

    def convert_reshape(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reshape.name

    def convert_flatten(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reshape.name

    def remove_node(self, node):
        input_name = node.inputs[0]
        output_name = node.outputs[0]
        self._replace_tensors[output_name] = input_name

    def convert_eltwise(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Eltwise.name
        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        type_arg.i = self.eltwise_type[node.op_type].value

        if node.op_type == OnnxOpType.Sqrt.name:
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = 0.5
        elif node.op_type == OnnxOpType.Reciprocal.name:
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = -1

    def convert_reduce(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reduce.name

        reduce_type_arg = op.arg.add()
        reduce_type_arg.name = MaceKeyword.mace_reduce_type_str
        reduce_type_arg.i = self.reduce_type[node.op_type].value

        if node.op_type in [OnnxOpType.GlobalAveragePool.name,
                            OnnxOpType.GlobalMaxPool.name]:
            reduce_dims = [2, 3]
            keep_dims = 1
        else:
            if 'axes' in node.attrs:
                reduce_dims = node.attrs['axes']
            else:
                reduce_dims = []
            if 'keepdims' in node.attrs:
                keep_dims = node.attrs['keepdims']
            else:
                keep_dims = 1
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.ints.extend(reduce_dims)

        keep_dims_arg = op.arg.add()
        keep_dims_arg.name = MaceKeyword.mace_keepdims_str
        keep_dims_arg.i = keep_dims

    def convert_imagescaler(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.BatchNorm.name

        scale = node.attrs['scale']
        bias_value = np.array(node.attrs['bias'])
        scale_value = scale * np.ones_like(bias_value)

        scale_name = node.name + "_scale"
        bias_name = node.name + "_bias"
        self.add_tensor(scale_name, scale_value.shape, mace_pb2.DT_FLOAT,
                        scale_value)
        self.add_tensor(bias_name, bias_value.shape, mace_pb2.DT_FLOAT,
                        bias_value)
        op.input.extend([scale_name, bias_name])

    def convert_matmul(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.MatMul.name

    def convert_softmax(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Softmax.name

    def convert_argmax(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.ArgMax.name

        if 'axis' in node.attrs:
            axis_value = node.attrs['axis']
        else:
            axis_value = 0
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis_value

        if 'keepdims' in node.attrs:
            keepdims = node.attrs['keepdims']
        else:
            keepdims = 1
        keep_dims_arg = op.arg.add()
        keep_dims_arg.name = MaceKeyword.mace_keepdims_str
        keep_dims_arg.i = keepdims

        if node.op_type == OnnxOpType.ArgMin.name:
            min_arg = op.arg.add()
            min_arg.name = MaceKeyword.mace_argmin_str
            min_arg.i = 1

    def convert_cast(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Cast.name

        if 'to' in node.attrs:
            dtype = node.attrs['to']
            if dtype == TensorProto.FLOAT:
                op.output_type.extend([self._option.data_type])
            elif dtype == TensorProto.INT:
                op.output_type.extend([mace_pb2.DT_INT32])
            else:
                mace_check(False, "data type %s not supported" % dtype)
        else:
            op.output_type.extend([self._option.data_type])

    def convert_depth_space(self, node):
        op = self.convert_general_op(node)
        if op.type == OnnxOpType.DepthToSpace.name:
            op.type = MaceOp.DepthToSpace.name
        else:
            op.type = MaceOp.SpaceToDepth.name
        mace_check(('block_size' in node.attrs),
                   "depth to space op should have block size attribute.")
        block_size = node.attrs['block_size']
        size_arg = op.arg.add()
        size_arg.name = MaceKeyword.mace_space_depth_block_size_str
        size_arg.i = block_size

    def convert_deconv(self, node):
        op = self.convert_general_op(node)

        self.add_stride_pad_kernel_arg(node.attrs, op)

        if 'group' in node.attrs:
            group_val = node.attrs["group"]
        else:
            group_val = 1
        if group_val > 1:
            op.type = MaceOp.DepthwiseDeconv2d.name
            filter_shape = self._graph_shapes_dict[node.inputs[1]]
            filter_tensor = self._consts[node.inputs[1]]
            new_shape = [filter_shape[1], filter_shape[0],
                         filter_shape[2], filter_shape[3]]
            del filter_tensor.dims[:]
            filter_tensor.dims.extend(new_shape)
        else:
            op.type = MaceOp.Deconv2D.name
        group_arg = op.arg.add()
        group_arg.name = MaceKeyword.mace_group_str
        group_arg.i = group_val

        dilation_arg = op.arg.add()
        dilation_arg.name = MaceKeyword.mace_dilations_str
        if 'dilations' in node.attrs:
            dilation_val = node.attrs["dilations"]
        else:
            dilation_val = [1, 1]
        dilation_arg.ints.extend(dilation_val)
        mace_check(dilation_val == [1, 1],
                   "not support convtranspose with dilation != 1 yet.")

        mace_check('output_padding' not in node.attrs,
                   "not support convtranspose with output_padding yet.")
        mace_check('output_shape' not in node.attrs,
                   "not support convtranspose with output_shape yet.")
        # TODO: if output shape specified, calculate padding value
        # if 'output_padding' in node.attrs:
        #     output_padding = node.attrs['output_padding']
        #     output_padding_arg = op.arg.add()
        #     output_padding_arg.name = MaceKeyword.mace_output_padding_str
        #     output_padding_arg.ints.extend(output_padding)
        # if 'output_shape' in node.attrs:
        #     output_shape = node.attrs['output_shape']
        #     output_shape_arg = op.arg.add()
        #     output_shape_arg.name = MaceKeyword.mace_output_shape_str
        #     output_shape_arg.ints.extend(output_shape)

    def convert_nop(self, node):
        pass

    def convert_identity(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Identity.name

    def convert_pad(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Pad.name

        if 'pads' in node.attrs:
            paddings_arg = op.arg.add()
            paddings_arg.name = MaceKeyword.mace_paddings_str
            paddings_value = node.attrs['pads']
            paddings_arg.ints.extend(paddings_value)

        if 'value' in node.attrs:
            constant_value_arg = op.arg.add()
            constant_value_arg.name = MaceKeyword.mace_constant_value_str
            constant_value_arg.i = node.attrs['value']

    def convert_gather(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Gather.name

        if 'axis' in node.attrs:
            value = node.attrs['axis']
        else:
            value = 0
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = value

    def convert_split(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Split.name

        if 'axis' in node.attrs:
            value = node.attrs['axis']
        else:
            value = 0
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = value

    def convert_transpose(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Transpose.name

        if np.array_equal(perm, ordered_perm):
            op.type = MaceOp.Identity.name
            del op.input[1:]
        if 'perm' in node.attrs:
            perm = node.attrs['perm']
            ordered_perm = np.sort(perm)
            if np.array_equal(perm, ordered_perm):
                op.type = MaceOp.Identity.name
            else:
                dims_arg = op.arg.add()
                dims_arg.name = MaceKeyword.mace_dims_str
                dims_arg.ints.extend(perm)

    @staticmethod
    def squeeze_shape(shape, axis):
        new_shape = []
        if len(axis) > 0:
            for i in range(len(shape)):
                if i not in axis:
                    new_shape.append(shape[i])
        else:
            new_shape = shape
        return new_shape

    def convert_squeeze(self, node):
        axis_value = node.attrs['axes']
        if node.inputs[0] in self._consts:
            tensor = self._consts[node.inputs[0]]
            shape = tensor.dims
            new_shape = self.squeeze_shape(shape, axis_value)
            del tensor.dims[:]
            tensor.dims.extend(new_shape)
            self.remove_node(node)
        else:
            op = self.convert_general_op(node)
            op.type = MaceOp.Squeeze.name
            axis_arg = op.arg.add()
            axis_arg.name = MaceKeyword.mace_axis_str
            if 'axis' in node.attrs:
                axis_value = node.attrs['axis']
            else:
                axis_value = []
            axis_arg.ints.extend(axis_value)

    @staticmethod
    def transpose_const(tensor):
        shape = tensor.dims
        mace_check(len(shape) == 2, "gemm only supports 2-dim input.")
        tensor_data = np.array(tensor.float_data).reshape(
            shape[0], shape[1])
        tensor_data = tensor_data.transpose(1, 0)
        tensor.float_data[:] = tensor_data.flat
        tensor.dims[:] = tensor_data.shape

    def convert_gemm(self, node):
        # only supports FullyConnected Style Gemm for now.
        trans_a = node.attrs['transA'] if 'transA' in node.attrs else 0
        trans_b = node.attrs['transB'] if 'transB' in node.attrs else 0
        shape_a = self._graph_shapes_dict[node.inputs[0]]
        shape_b = self._graph_shapes_dict[node.inputs[1]]
        mace_check(trans_a == 0 and trans_b == 1,
                   "Do not support non-default transpose")
        mace_check(len(shape_a) == 4,
                   "Unexpected fc input ndim.")
        mace_check(node.inputs[1] in self._consts, "unexpect fc weight.")
        if len(shape_b) == 4:
            mace_check(list(shape_b[2:]) == [1, 1],
                       "Only support 4D weight with shape [*, *, 1, 1]")
        elif len(shape_b) == 2:
            tensor_b = self._consts[node.inputs[1]]
            tensor_data = np.array(tensor_b.float_data).reshape(
                    shape_b[0], shape_b[1], 1, 1)
            tensor_b.float_data[:] = tensor_data.flat
            tensor_b.dims[:] = tensor_data.shape
        else:
            mace_check(False, "Unexpected fc weigth ndim.")

        op = self._mace_net_def.op.add()
        op.name = node.name
        op.type = MaceOp.FullyConnected.name
        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        data_type_arg.i = self._option.data_type

        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.ONNX.value

        ConverterUtil.add_data_format_arg(op, DataFormat.NCHW)

        for input in node.inputs:
            op.input.append(input)
        for output in node.outputs:
            op.output.append(output)
            output_shape = op.output_shape.add()
            shape_info = self._graph_shapes_dict[output]
            mace_check(len(shape_info) in [2, 4],
                       "gemm output shape should be 2 or 4 dims.")
            if len(shape_info) == 4:
                mace_check(shape_info[2] == 1 and shape_info[3] == 1,
                           "gemm's 4-dim output shape should be [*, * , 1, 1]")
            else:
                shape_info = [shape_info[0], shape_info[1], 1, 1]
            output_shape.dims.extend(shape_info)

        return op
