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
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.convert_util import mace_check

import numpy as np

import onnx
import onnx.utils
from onnx import mapping, numpy_helper, TensorProto
from numbers import Number

IS_PYTHON3 = sys.version_info > (3,)


class AttributeType(Enum):
    INT = 100
    FLOAT = 101
    INTS = 102
    FLOATS = 103
    BOOL = 104


OnnxSupportedOps = [
    'Abs',
    # 'Acos',
    # 'Acosh',
    'Add',
    'Affine',
    # 'And',
    'Append',
    'ArgMax',
    'ArgMin',
    # 'Asin',
    # 'Asinh',
    # 'Atan',
    # 'Atanh',
    'AveragePool',
    'BatchNormalization',
    'BatchNorm',
    'Cast',
    # 'Ceil',
    'Clip',
    # 'Compress',
    'Concat',
    # 'Constant',
    # 'ConstantLike',
    'Conv',
    'ConvTranspose',
    # 'Cos',
    # 'Cosh',
    'DepthToSpace',
    'DimRange',
    'Div',
    'Dropout',
    'DynamicLSTM',
    'Elu',
    'Equal',
    # 'Exp',
    # 'Expand',
    'ExtractPooling',
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
    'IfDefined',
    'ImageScaler',
    # 'InstanceNormalization',
    # 'LRN',
    'LSTM',
    'LstmNonlinear',
    'LeakyRelu',
    # 'Less',
    # 'Log',
    'LogSoftmax',
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
    'Normalize',
    # 'Not',
    'Offset',
    # 'OneHot',
    # 'Or',
    'PRelu',
    'Pad',
    'PadContext',
    'PNorm',
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
    'Scale',
    # 'Scan',
    # 'Selu',
    'Shape',
    'Sigmoid',
    # 'Sin',
    # 'Sinh',
    # 'Size',
    'Slice',
    'Softmax',
    # 'Softplus',
    # 'Softsign',
    'SpaceToDepth',
    'Splice',
    'Split',
    'Sqrt',
    'Squeeze',
    'Sub',
    'Sum',
    'SumGroup',
    # 'Tan',
    'Tanh',
    'TargetRMSNorm',
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
            if IS_PYTHON3 else attr_proto.s
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
        if self.name == '':
            self.name = str(node.output)
        self.op_type = str(node.op_type)
        self.domain = str(node.domain)
        self.attrs = dict([(attr.name,
                            translate_onnx(attr.name, convert_onnx(attr)))
                           for attr in node.attribute])
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.node_proto = node

    def print_info(self):
        print("node: ", self.name)
        print("    type: ", self.op_type)
        print("    domain: ", self.domain)
        print("    inputs: ", self.inputs)
        print("    outputs: ", self.outputs)
        print("    attrs:")
        for arg in self.attrs:
            print("        %s: %s" % (arg, self.attrs[arg]))


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
        OnnxOpType.Scale.name: EltwiseType.PROD,
        OnnxOpType.Clip.name: EltwiseType.CLIP,
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
            OnnxOpType.Affine.name: self.convert_affine,
            OnnxOpType.Append.name: self.convert_concat,
            OnnxOpType.ArgMax.name: self.convert_argmax,
            OnnxOpType.ArgMin.name: self.convert_argmax,
            OnnxOpType.AveragePool.name: self.convert_pooling,
            OnnxOpType.BatchNormalization.name: self.convert_fused_batchnorm,
            OnnxOpType.BatchNorm.name: self.convert_fused_batchnorm,
            OnnxOpType.Cast.name: self.convert_cast,
            OnnxOpType.Clip.name: self.convert_eltwise,
            OnnxOpType.Concat.name: self.convert_concat,
            OnnxOpType.Conv.name: self.convert_conv2d,
            OnnxOpType.ConvTranspose.name: self.convert_deconv,
            OnnxOpType.DepthToSpace.name: self.convert_depth_space,
            OnnxOpType.Dropout.name: self.convert_identity,
            OnnxOpType.DimRange.name: self.convert_dim_range,
            OnnxOpType.Div.name: self.convert_eltwise,
            OnnxOpType.Equal.name: self.convert_eltwise,
            OnnxOpType.ExtractPooling.name: self.convert_extract_pooling,
            OnnxOpType.Gather.name: self.convert_gather,
            OnnxOpType.Gemm.name: self.convert_gemm,
            OnnxOpType.GlobalAveragePool.name: self.convert_reduce,
            OnnxOpType.GlobalMaxPool.name: self.convert_reduce,
            OnnxOpType.Identity.name: self.convert_identity,
            OnnxOpType.IfDefined.name: self.convert_ifdefined,
            OnnxOpType.ImageScaler.name: self.convert_imagescaler,
            OnnxOpType.LeakyRelu.name: self.convert_activation,
            OnnxOpType.LogSoftmax.name: self.convert_softmax,
            OnnxOpType.LstmNonlinear.name: self.convert_lstm_nonlinear,
            OnnxOpType.DynamicLSTM.name: self.convert_dynamic_lstm,
            OnnxOpType.Max.name: self.convert_eltwise,
            OnnxOpType.MaxPool.name: self.convert_pooling,
            OnnxOpType.MatMul.name: self.convert_matmul,
            OnnxOpType.Min.name: self.convert_eltwise,
            OnnxOpType.Mul.name: self.convert_eltwise,
            OnnxOpType.Neg.name: self.convert_eltwise,
            OnnxOpType.Normalize: self.convert_normalize,
            OnnxOpType.Offset.name: self.convert_identity,
            OnnxOpType.Pad.name: self.convert_pad,
            OnnxOpType.PadContext.name: self.convert_pad_context,
            OnnxOpType.PNorm.name: self.convert_pnorm,
            OnnxOpType.Pow.name: self.convert_eltwise,
            OnnxOpType.PRelu.name: self.convert_activation,
            OnnxOpType.Relu.name: self.convert_activation,
            OnnxOpType.Reshape.name: self.convert_reshape,
            OnnxOpType.Reciprocal.name: self.convert_eltwise,
            OnnxOpType.Scale.name: self.convert_eltwise,
            OnnxOpType.Sigmoid.name: self.convert_activation,
            OnnxOpType.Slice.name: self.convert_slice,
            OnnxOpType.Softmax.name: self.convert_softmax,
            OnnxOpType.SpaceToDepth.name: self.convert_depth_space,
            OnnxOpType.Splice.name: self.convert_splice,
            OnnxOpType.Split.name: self.convert_split,
            OnnxOpType.Sqrt.name: self.convert_eltwise,
            OnnxOpType.Squeeze.name: self.convert_squeeze,
            OnnxOpType.Sub.name: self.convert_eltwise,
            OnnxOpType.Sum.name: self.convert_eltwise,
            OnnxOpType.SumGroup.name: self.convert_sum_group,
            OnnxOpType.Tanh.name: self.convert_activation,
            OnnxOpType.TargetRMSNorm: self.convert_target_rms_norm,
            OnnxOpType.Transpose.name: self.convert_transpose,
        }
        self._option = option
        self._mace_net_def = mace_pb2.NetDef()
        self._data_format = DataFormat.NCHW
        ConverterUtil.set_filter_format(self._mace_net_def, DataFormat.OIHW)
        ConverterUtil.add_data_format_arg(self._mace_net_def,
                                          self._data_format)
        onnx_model = onnx.load(src_model_file)

        ir_version = onnx_model.ir_version
        opset_imp = onnx_model.opset_import

        self._isKaldi = False

        polish_available = True
        print("onnx model IR version: ", ir_version)
        for imp in opset_imp:
            domain = imp.domain
            version = imp.version
            print("constains ops domain: ", domain, "version:", version)
            if 'kaldi2onnx' in domain:
                polish_available = False
                self._data_format = DataFormat.NONE
                self._isKaldi = True
        if polish_available:
            onnx_model = onnx.utils.polish_model(onnx_model)

        self._onnx_model = onnx_model
        self._graph_shapes_dict = {}
        self._consts = {}
        self._replace_tensors = {}

    @staticmethod
    def print_graph_info(graph):
        for value_info in graph.value_info:
            print("value info:", value_info)
        for value_info in graph.input:
            print("inputs info:", value_info)
        for value_info in graph.output:
            print("outputs info:", value_info)

    def extract_shape_info(self, graph):
        def extract_value_info(shape_dict, value_info):
            t = tuple([int(dim.dim_value)
                       for dim in value_info.type.tensor_type.shape.dim])
            if t:
                shape_dict[value_info.name] = t

        for vi in graph.value_info:
            extract_value_info(self._graph_shapes_dict, vi)
        for vi in graph.input:
            extract_value_info(self._graph_shapes_dict, vi)
        for vi in graph.output:
            extract_value_info(self._graph_shapes_dict, vi)

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

    def remove_node(self, node):
        input_name = node.inputs[0]
        output_name = node.outputs[0]
        self._replace_tensors[output_name] = input_name

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

    @staticmethod
    def transpose_const(tensor):
        shape = tensor.dims
        mace_check(len(shape) == 2, "gemm only supports 2-dim input.")
        tensor_data = np.array(tensor.float_data).reshape(
            shape[0], shape[1])
        tensor_data = tensor_data.transpose(1, 0)
        tensor.float_data[:] = tensor_data.flat
        tensor.dims[:] = tensor_data.shape

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

    def convert_general_op(self, node, with_shape=True):
        op = self._mace_net_def.op.add()
        op.name = node.name

        for input in node.inputs:
            if input in self._replace_tensors:
                input = self._replace_tensors[input]
            op.input.append(input)
        for output in node.outputs:
            op.output.append(output)
            if with_shape:
                if output in self._graph_shapes_dict:
                    output_shape = op.output_shape.add()
                    shape_info = self._graph_shapes_dict[output]
                    output_shape.dims.extend(shape_info)

        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        data_type_arg.i = self._option.data_type

        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.ONNX.value

        ConverterUtil.add_data_format_arg(op, self._data_format)
        return op

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

    def convert_affine(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.MatMul.name
        transpose_b_arg = op.arg.add()
        transpose_b_arg.name = MaceKeyword.mace_transpose_b_str
        transpose_b_arg.i = 1

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

    def convert_biasadd(self, node):
        self.convert_general_op(node)
        op.type = MaceOp.BiasAdd.name

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

    def convert_concat(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Concat.name
        axis_value = 1
        if node.op_type == OnnxOpType.Concat.name:
            mace_check('axis' in node.attrs,
                       'Concat op should have axis attribute.')
            axis_value = node.attrs['axis']
            mace_check(axis_value == 1 or axis_value == -3,
                       "only support concat at channel dimension")
        elif node.op_type == OnnxOpType.Append.name:
            axis_value = -1
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis_value

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
            mace_check(op.input[1] in self._consts,
                       "Mace does not support non-const filter convolution.")

        dilation_arg = op.arg.add()
        dilation_arg.name = MaceKeyword.mace_dilations_str
        if 'dilations' in node.attrs:
            dilation_val = node.attrs["dilations"]
        else:
            dilation_val = [1, 1]
        dilation_arg.ints.extend(dilation_val)

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

    def convert_dim_range(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Slice.name

        mace_check('offset' in node.attrs,
                   "Attribute dim required!")
        mace_check('output_dim' in node.attrs,
                   "Attribute output_dim required!")
        offset = node.attrs['offset']
        starts_arg = op.arg.add()
        starts_arg.name = 'starts'
        starts_arg.ints.extend([offset])
        output_dim = node.attrs['output_dim']
        ends_arg = op.arg.add()
        ends_arg.name = 'ends'
        ends_arg.ints.extend([output_dim + offset])
        axes_arg = op.arg.add()
        axes_arg.name = 'axes'
        axes_arg.ints.extend([-1])

    def convert_dynamic_lstm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.DynamicLSTM.name

        if 'prev_out_delay' in node.attrs:
            prev_out_delay = node.attrs['prev_out_delay']
            mace_check(prev_out_delay < 0,
                       "dynamic's prev_out_delay should <= 0.")
            prev_out_delay_arg = op.arg.add()
            prev_out_delay_arg.name = 'prev_out_delay'
            prev_out_delay_arg.i = prev_out_delay
        if 'prev_cell_delay' in node.attrs:
            prev_cell_delay = node.attrs['prev_cell_delay']
            mace_check(prev_cell_delay < 0,
                       "dynamic's prev_cell_delay should < 0.")
            prev_cell_delay_arg = op.arg.add()
            prev_cell_delay_arg.name = 'prev_cell_delay'
            prev_cell_delay_arg.i = prev_cell_delay
        if 'prev_out_offset' in node.attrs:
            prev_out_offset = node.attrs['prev_out_offset']
            mace_check(prev_out_offset >= 0,
                       "dynamic's prev_out_offset should >= 0.")
            prev_out_offset_arg = op.arg.add()
            prev_out_offset_arg.name = 'prev_out_offset'
            prev_out_offset_arg.i = prev_out_offset
        if 'prev_out_dim' in node.attrs:
            prev_out_dim = node.attrs['prev_out_dim']
            mace_check(prev_out_dim > 0,
                       "dynamic's prev_out_dim should > 0.")
            prev_out_dim_arg = op.arg.add()
            prev_out_dim_arg.name = 'prev_out_dim'
            prev_out_dim_arg.i = prev_out_dim
        if 'prev_cell_dim' in node.attrs:
            prev_cell_dim = node.attrs['prev_cell_dim']
            mace_check(prev_cell_dim > 0,
                       "dynamic's prev_cell_dim should > 0.")
            prev_cell_dim_arg = op.arg.add()
            prev_cell_dim_arg.name = 'prev_cell_dim'
            prev_cell_dim_arg.i = prev_cell_dim
        if 'bias_a' in node.attrs:
            bias_a = node.attrs['bias_a']
            bias_a_arg = op.arg.add()
            bias_a_arg.name = 'bias_a'
            bias_a_arg.i = bias_a
        if 'bias_b' in node.attrs:
            bias_b = node.attrs['bias_b']
            bias_b_arg = op.arg.add()
            bias_b_arg.name = 'bias_b'
            bias_b_arg.i = bias_b
        if 'scale' in node.attrs:
            scale = node.attrs['scale']
            scale_arg = op.arg.add()
            scale_arg.name = 'scale'
            scale_arg.f = scale

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
        elif node.op_type == OnnxOpType.Scale.name and 'scale' in node.attrs:
            value = node.attrs['scale']
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = value
        elif node.op_type == OnnxOpType.Clip.name:
            if 'min' in node.attrs:
                min_value = node.attrs['min']
            else:
                min_value = np.finfo(np.float32).min
            if 'max' in node.attrs:
                max_value = node.attrs['max']
            else:
                max_value = np.finfo(np.float32).max
            coeff_arg = op.arg.add()
            coeff_arg.name = MaceKeyword.mace_coeff_str
            coeff_arg.floats.extend([min_value, max_value])

    @staticmethod
    def copy_node_attr(op, node, attr_name, dtype=AttributeType.INT,
                       default=None):
        if attr_name in node.attrs or default is not None:
            if attr_name in node.attrs:
                value = node.attrs[attr_name]
            else:
                value = default
            new_arg = op.arg.add()
            new_arg.name = attr_name
            if dtype == AttributeType.INT:
                new_arg.i = int(value)
            elif dtype == AttributeType.FLOAT:
                new_arg.f = float(value)
            elif dtype == AttributeType.INTS:
                new_arg.ints.extend(value)
            elif dtype == AttributeType.FLOATS:
                new_arg.floats.extend(value)
            return value
        else:
            return default

    def convert_extract_pooling(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.ExtractPooling.name

        self.copy_node_attr(op, node, 'include_variance', AttributeType.INT)
        self.copy_node_attr(op, node, 'num_log_count', AttributeType.INT)
        self.copy_node_attr(op, node, 'variance_floor', AttributeType.FLOAT)
        self.copy_node_attr(op, node, 'input_time_range', AttributeType.INTS)
        self.copy_node_attr(op, node, 'input_indexes', AttributeType.INTS)

        if 'output_time_range' in node.attrs:
            output_time_range = node.attrs['output_time_range']
            mace_check(len(output_time_range) == 2,
                       "output time range should have two values.")
            out_start_index = output_time_range[0]
            out_end_index = output_time_range[1]
        else:
            mace_check('start_index' in node.attrs and
                       'end_index' in node.attrs,
                       "'start_index' and 'end_index'"
                       " are required in ExtractPooling.")
            out_start_index = node.attrs['start_index']
            out_end_index = node.attrs['end_index']
            output_time_range = [out_start_index, out_end_index]

        output_time_range_arg = op.arg.add()
        output_time_range_arg.name = 'output_time_range'
        output_time_range_arg.ints.extend(output_time_range)

        mace_check('modulus' in node.attrs,
                   "'modulus' is required in ExtractPooling.")
        mace_check('output_indexes' in node.attrs,
                   "'output_indexes' is required in ExtractPooling.")
        mace_check('counts' in node.attrs,
                   "'counts' is required in ExtractPooling.")
        mace_check('forward_indexes' in node.attrs,
                   "'forward_indexes' is required in ExtractPooling.")
        modulus = node.attrs['modulus']
        output_indexes = node.attrs['output_indexes']
        counts = node.attrs['counts']
        forward_indexes = node.attrs['forward_indexes']

        mace_check(len(counts) == len(output_indexes) and
                   len(forward_indexes) == 2 * len(output_indexes),
                   "output_indexes length:%s "
                   "counts length:%s "
                   "forward_indexes length:%s"
                   % (len(output_indexes), len(counts), len(forward_indexes)))

        new_output_indexes = []
        new_forward_indexes = []
        new_counts = []
        for i in range(len(output_indexes)):
            if output_indexes[i] + modulus > out_start_index and\
                    output_indexes[i] <= out_end_index:
                new_output_indexes.append(output_indexes[i])
                new_counts.append(counts[i])
                new_forward_indexes.append(forward_indexes[2 * i])
                new_forward_indexes.append(forward_indexes[2 * i + 1])
        modulus_arg = op.arg.add()
        modulus_arg.name = 'modulus'
        modulus_arg.i = modulus

        counts_arg = op.arg.add()
        counts_arg.name = 'counts'
        counts_arg.floats.extend(new_counts)

        forward_indexes_arg = op.arg.add()
        forward_indexes_arg.name = 'forward_indexes'
        forward_indexes_arg.ints.extend(new_forward_indexes)

        output_indexes_arg = op.arg.add()
        output_indexes_arg.name = 'output_indexes'
        output_indexes_arg.ints.extend(new_output_indexes)

    def convert_flatten(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reshape.name

    def convert_kaldi_batchnorm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.KaldiBatchNorm.name
        dim = self.copy_node_attr(op, node,
                                  'dim', AttributeType.INT, -1)
        block_dim = self.copy_node_attr(op, node,
                                        'block_dim',
                                        AttributeType.INT, -1)
        epsilon = self.copy_node_attr(op, node,
                                      'epsilon',
                                      AttributeType.FLOAT, 1e-3)
        target_rms = self.copy_node_attr(op, node,
                                         'target_rms',
                                         AttributeType.FLOAT, 1.0)
        test_mode = self.copy_node_attr(op, node,
                                        'test_mode',
                                        AttributeType.INT, 0)
        mace_check(block_dim > 0 and
                   dim % block_dim == 0 and
                   epsilon > 0 and
                   target_rms > 0, "attributes invalid.")

        if test_mode > 0:
            mace_check(len(node.inputs) == 3,
                       "Kaldi's BatchNorm should have 3 inputs.")
            stats_mean = np.array(self._consts[node.inputs[1]].float_data)
            stats_var = np.array(self._consts[node.inputs[2]].float_data)
            offset_value = -1.0 * stats_mean
            scale_value = stats_var
            scale_value[scale_value < 0] = 0
            scale_value = np.power(scale_value + epsilon, -0.5) * target_rms
            offset_value = offset_value * scale_value
            scale_name = node.name + '_scale'
            offset_name = node.name + '_offset'
            self.add_tensor(scale_name, scale_value.shape,
                            mace_pb2.DT_FLOAT, scale_value)
            self.add_tensor(offset_name, offset_value.shape,
                            mace_pb2.DT_FLOAT, offset_value)
            del op.input[1:]
            op.input.extend([scale_name, offset_name])
            del op.output[1:]
            del op.output_shape[1:]

    def convert_fused_batchnorm(self, node):
        if self._isKaldi:
            self.convert_kaldi_batchnorm(node)
            return
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

    def convert_identity(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Identity.name

    def convert_ifdefined(self, node):
        op = self.convert_general_op(node)
        if 'offset' in node.attrs:
            offset = node.attrs['offset']
        else:
            offset = 0
        mace_check(offset <= 0, "IfDefined's offset should be <= 0.")
        if offset == 0:
            op.type = MaceOp.Identity.name
        else:
            op.type = MaceOp.Delay.name
        offset_arg = op.arg.add()
        offset_arg.name = 'offset'
        offset_arg.i = node.attrs['offset']

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

    def convert_lstm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.LSTMCell.name

    def convert_lstm_nonlinear(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.LstmNonlinear.name

    def convert_matmul(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.MatMul.name

    def convert_nop(self, node):
        pass

    def convert_normalize(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.BatchNorm.name

    def convert_pad(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Pad.name
        if 'mode' in node.attrs:
            mode = node.attrs['mode']
            padding_type_arg = op.arg.add()
            padding_type_arg.name = MaceKeyword.mace_padding_type_str
            if mode == 'reflect':
                padding_type_arg.i = PadType.REFLECT
            elif mode == 'edge':
                padding_type_arg.i = PadType.SYMMETRIC
            else:
                padding_type_arg.i = PadType.CONSTANT
        if 'pads' in node.attrs:
            paddings_arg = op.arg.add()
            paddings_arg.name = MaceKeyword.mace_paddings_str
            paddings_value = node.attrs['pads']
            paddings_arg.ints.extend(paddings_value)
        if 'value' in node.attrs:
            constant_value_arg = op.arg.add()
            constant_value_arg.name = MaceKeyword.mace_constant_value_str
            constant_value_arg.f = node.attrs['value']

    def convert_pad_context(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.PadContext.name
        if 'left_context' in node.attrs:
            left_context_arg = op.arg.add()
            left_context_arg.name = 'left_context'
            left_context_arg.i = node.attrs['left_context']
        if 'right_context' in node.attrs:
            right_context_arg = op.arg.add()
            right_context_arg.name = 'right_context'
            right_context_arg.i = node.attrs['right_context']

    def convert_pnorm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.PNorm.name
        if 'output_dim' in node.attrs:
            output_dim_arg = op.arg.add()
            output_dim_arg.name = 'output_dim'
            output_dim_arg.i = node.attrs['output_dim']
        if 'p' in node.attrs:
            p_value = node.attrs['p']
            mace_check((p_value >= 0) and (p_value <= 2),
                       "PNorm only supports p = 0, 1, 2")
            p_arg = op.arg.add()
            p_arg.name = 'p'
            p_arg.i = p_value

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

    def convert_reshape(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reshape.name

    def convert_slice(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Slice.name

        mace_check('starts' in node.attrs, "Attribute starts required!")
        mace_check('ends' in node.attrs, "Attribute ends required!")
        starts = node.attrs['starts']
        starts_arg = op.arg.add()
        starts_arg.name = 'starts'
        starts_arg.ints.extend(starts)
        ends = node.attrs['ends']
        ends_arg = op.arg.add()
        ends_arg.name = 'ends'
        ends_arg.ints.extend(ends)
        if 'axes' in node.attrs:
            axes = node.attrs['axes']
            axes_arg = op.arg.add()
            axes_arg.name = 'axes'
            axes_arg.ints.extend(axes)

    def convert_softmax(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Softmax.name
        if node.op_type == OnnxOpType.LogSoftmax.name:
            use_log_arg = op.arg.add()
            use_log_arg.name = 'use_log'
            use_log_arg.i = 1

    def convert_splice(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Splice.name
        if 'context' in node.attrs:
            context = node.attrs['context']
        else:
            context = [0]
        context_arg = op.arg.add()
        context_arg.name = 'context'
        context_arg.ints.extend(context)
        if 'const_component_dim' in node.attrs:
            const_dim = node.attrs['const_component_dim']
        else:
            const_dim = 0
        const_dim_arg = op.arg.add()
        const_dim_arg.name = 'const_component_dim'
        const_dim_arg.i = const_dim

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

    def convert_sum_group(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.SumGroup.name

    def convert_target_rms_norm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.TargetRMSNorm.name

        self.copy_node_attr(op, node, 'target_rms', AttributeType.FLOAT)
        self.copy_node_attr(op, node, 'add_log_stddev', AttributeType.INT,
                            default=0)
        self.copy_node_attr(op, node, 'block_dim', AttributeType.INT,
                            default=0)

    def convert_transpose(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Transpose.name

        if 'perm' in node.attrs:
            perm = node.attrs['perm']
            ordered_perm = np.sort(perm)
            if np.array_equal(perm, ordered_perm):
                op.type = MaceOp.Identity.name
                del op.input[1:]
            else:
                dims_arg = op.arg.add()
                dims_arg.name = MaceKeyword.mace_dims_str
                dims_arg.ints.extend(perm)

    def convert_timeoffset(self, node):
        op = self.convert_general_op(node)
        mace_check('offset' in node.attrs,
                   'Offset attribute required in Offset Node.')
        offset = node.attrs['offset']
        if offset == 0:
            op.type = MaceOp.Identity.name
        else:
            op.type = MaceOp.TimeOffset.name

        chunk_size = node.attrs['chunk_size']
        chunk_size_arg = op.arg.add()
        chunk_size_arg.name = 'chunk_size'
        chunk_size_arg.i = chunk_size

        offset_arg = op.arg.add()
        offset_arg.name = 'offset'
        offset_arg.i = offset
