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

from py_proto import mace_pb2
from transform import base_converter
from transform import shape_inference
from transform.base_converter import ActivationType
from transform.base_converter import ConverterUtil
from transform.base_converter import CoordinateTransformationMode
from transform.base_converter import DataFormat
from transform.base_converter import EltwiseType
from transform.base_converter import FrameworkType
from transform.base_converter import InfoKey
from transform.base_converter import MaceOp
from transform.base_converter import MaceKeyword
from transform.base_converter import PoolingType
from transform.base_converter import PaddingMode
from transform.base_converter import PadType
from transform.base_converter import QatType
from transform.base_converter import ReduceType
from transform.base_converter import RoundMode

from quantize import quantize_util
from utils.util import mace_check
from utils.util import MaceLogger

import numpy as np

import onnx
import onnx.utils
from onnx import mapping, numpy_helper, shape_inference, TensorProto
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
    'Constant',
    # 'ConstantLike',
    'Conv',
    'ConvTranspose',
    # 'Cos',
    # 'Cosh',
    'DepthToSpace',
    'DequantizeLinear',
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
    'Flatten',
    # 'Floor',
    # 'GRU',
    'Gather',
    'Gemm',
    'GlobalAveragePool',
    # 'GlobalLpPool',
    'GlobalMaxPool',
    # 'Greater',
    'HardSigmoid',
    # 'Hardmax',
    'Identity',
    # 'If',
    'IfDefined',
    'ImageScaler',
    'InstanceNormalization',
    # 'LRN',
    'Linear',
    'LSTM',
    'LstmNonlinear',
    'LeakyRelu',
    # 'Less',
    # 'Log',
    'LogSoftmax',
    # 'Loop',
    'LpNormalization',
    # 'LpPool',
    'MatMul',
    'Max',
    'MaxPool',
    # 'MaxRoiPool',
    # 'MaxUnpool',
    # 'Mean',
    'Min',
    'Mul',
    # 'Multinomial',
    'Neg',
    'NoOp',
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
    'QuantizeLinear',
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
    'ReduceSum',
    # 'ReduceSumSquare',
    'Relu',
    'ReplaceIndex',
    'Resize',
    'Reshape',
    'Round',
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
    'Subsample',
    'Sum',
    'SumGroup',
    # 'Tan',
    'Tanh',
    'TargetRMSNorm',
    # 'Tile',
    # 'TopK',
    'Transpose',
    'Where',
    'Unsqueeze',
    'Upsample',
    # 'Xor',
]

OnnxOpType = Enum('OnnxOpType',
                  [(op, op) for op in OnnxSupportedOps],
                  type=str)

onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: onnx_dtype(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: onnx_dtype(x),
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
            self.name = str('_'.join(node.output))
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
        OnnxOpType.GlobalAveragePool.name: PoolingType.AVG,
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
        OnnxOpType.ReduceSum.name: ReduceType.SUM,
    }

    activation_type = {
        OnnxOpType.Relu.name: ActivationType.RELU,
        OnnxOpType.LeakyRelu.name: ActivationType.LEAKYRELU,
        OnnxOpType.PRelu.name: ActivationType.PRELU,
        OnnxOpType.Elu.name: ActivationType.ELU,
        OnnxOpType.Tanh.name: ActivationType.TANH,
        OnnxOpType.Sigmoid.name: ActivationType.SIGMOID,
        OnnxOpType.HardSigmoid.name: ActivationType.HARDSIGMOID,
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
            OnnxOpType.Clip.name: self.convert_clip,
            OnnxOpType.Concat.name: self.convert_concat,
            OnnxOpType.Conv.name: self.convert_conv2d,
            OnnxOpType.ConvTranspose.name: self.convert_deconv,
            OnnxOpType.Constant.name: self.convert_constant,
            OnnxOpType.DepthToSpace.name: self.convert_depth_space,
            OnnxOpType.DequantizeLinear.name: self.convert_dequantize_linear,
            OnnxOpType.Dropout.name: self.convert_dropout,
            OnnxOpType.DimRange.name: self.convert_dim_range,
            OnnxOpType.Div.name: self.convert_eltwise,
            OnnxOpType.Elu.name: self.convert_activation,
            OnnxOpType.Equal.name: self.convert_eltwise,
            OnnxOpType.ExtractPooling.name: self.convert_extract_pooling,
            OnnxOpType.Flatten.name: self.convert_flatten,
            OnnxOpType.Gather.name: self.convert_gather,
            OnnxOpType.Gemm.name: self.convert_gemm,
            OnnxOpType.GlobalAveragePool.name: self.convert_reduce,
            OnnxOpType.GlobalMaxPool.name: self.convert_reduce,
            OnnxOpType.HardSigmoid.name: self.convert_activation,
            OnnxOpType.Identity.name: self.convert_identity,
            OnnxOpType.IfDefined.name: self.convert_ifdefined,
            OnnxOpType.ImageScaler.name: self.convert_imagescaler,
            OnnxOpType.InstanceNormalization.name: self.convert_instance_norm,
            OnnxOpType.LeakyRelu.name: self.convert_activation,
            OnnxOpType.Linear.name: self.convert_affine,
            OnnxOpType.LogSoftmax.name: self.convert_softmax,
            OnnxOpType.LpNormalization: self.convert_lpnormalization,
            OnnxOpType.LstmNonlinear.name: self.convert_lstm_nonlinear,
            OnnxOpType.DynamicLSTM.name: self.convert_dynamic_lstm,
            OnnxOpType.Max.name: self.convert_eltwise,
            OnnxOpType.MaxPool.name: self.convert_pooling,
            OnnxOpType.MatMul.name: self.convert_matmul,
            OnnxOpType.Min.name: self.convert_eltwise,
            OnnxOpType.Mul.name: self.convert_eltwise,
            OnnxOpType.Neg.name: self.convert_eltwise,
            OnnxOpType.NoOp.name: self.convert_identity,
            OnnxOpType.Normalize: self.convert_normalize,
            OnnxOpType.Offset.name: self.convert_subsample,
            OnnxOpType.Pad.name: self.convert_pad,
            OnnxOpType.PadContext.name: self.convert_pad_context,
            OnnxOpType.PNorm.name: self.convert_pnorm,
            OnnxOpType.Pow.name: self.convert_eltwise,
            OnnxOpType.PRelu.name: self.convert_activation,
            OnnxOpType.QuantizeLinear.name: self.convert_quantize_linear,
            OnnxOpType.Relu.name: self.convert_activation,
            OnnxOpType.Reshape.name: self.convert_reshape,
            OnnxOpType.Reciprocal.name: self.convert_eltwise,
            OnnxOpType.ReduceMax.name: self.convert_reduce,
            OnnxOpType.ReduceMean.name: self.convert_reduce,
            OnnxOpType.ReduceMin.name: self.convert_reduce,
            OnnxOpType.ReduceProd.name: self.convert_reduce,
            OnnxOpType.ReduceSum.name: self.convert_reduce,
            OnnxOpType.ReplaceIndex.name: self.convert_replaceindex,
            OnnxOpType.Round.name: self.convert_replaceindex,
            OnnxOpType.Scale.name: self.convert_eltwise,
            OnnxOpType.Shape.name: self.convert_shape,
            OnnxOpType.Sigmoid.name: self.convert_activation,
            OnnxOpType.Slice.name: self.convert_slice,
            OnnxOpType.Softmax.name: self.convert_softmax,
            OnnxOpType.SpaceToDepth.name: self.convert_depth_space,
            OnnxOpType.Splice.name: self.convert_splice,
            OnnxOpType.Split.name: self.convert_split,
            OnnxOpType.Sqrt.name: self.convert_eltwise,
            OnnxOpType.Squeeze.name: self.convert_squeeze,
            OnnxOpType.Sub.name: self.convert_eltwise,
            OnnxOpType.Subsample.name: self.convert_subsample,
            OnnxOpType.Sum.name: self.convert_eltwise,
            OnnxOpType.SumGroup.name: self.convert_sum_group,
            OnnxOpType.Tanh.name: self.convert_activation,
            OnnxOpType.TargetRMSNorm: self.convert_target_rms_norm,
            OnnxOpType.Transpose.name: self.convert_transpose,
            OnnxOpType.Unsqueeze.name: self.convert_unsqueeze,
            OnnxOpType.Upsample.name: self.convert_upsample,
            OnnxOpType.Resize.name: self.convert_resize,
            OnnxOpType.Where.name: self.convert_where
        }
        self._option = option
        self._converter_info = dict()
        self._converter_info[InfoKey.qat_type] = dict()
        self._converter_info[InfoKey.has_qat] = set()
        self._mace_net_def = mace_pb2.NetDef()
        self._data_format = DataFormat.NCHW
        ConverterUtil.set_filter_format(self._mace_net_def, DataFormat.OIHW)
        ConverterUtil.add_data_format_arg(self._mace_net_def,
                                          self._data_format)
        ConverterUtil.set_framework_type(
            self._mace_net_def, FrameworkType.ONNX.value)
        onnx_model = onnx.load(src_model_file)

        ir_version = onnx_model.ir_version
        opset_imp = onnx_model.opset_import

        onnx.checker.check_model(onnx_model)

        self._isKaldi = False

        polish_available = True
        print("onnx model IR version: ", ir_version)
        for imp in opset_imp:
            domain = imp.domain
            version = imp.version
            print("constains ops domain: ", domain, "version:", version)
            if 'kaldi' in domain:
                polish_available = False
                self._data_format = DataFormat.NONE
                self._isKaldi = True
        if polish_available and hasattr(onnx.utils, "polish_model"):
            onnx_model = onnx.utils.polish_model(onnx_model)

        self._opset_version = onnx_model.opset_import[0].version
        self._onnx_model = onnx_model
        self._graph_shapes_dict = {}
        self._consts = {}
        self._replace_tensors = {}
        self._source_framework = self._onnx_model.producer_name
        self._consumers = {}
        self._producer = {}
        self.construct_producer_consumers()
        self._scale_zeros = {}
        self._skip_quant_dequants = set()

    @staticmethod
    def print_graph_info(graph):
        for value_info in graph.value_info:
            print("value info:", value_info)
        for value_info in graph.input:
            print("inputs info:", value_info)
        for value_info in graph.output:
            print("outputs info:", value_info)

    def construct_producer_consumers(self):
        for node in self._onnx_model.graph.node:
            for input_tensor in node.input:
                if input_tensor not in self._consumers:
                    self._consumers[input_tensor] = []
                self._consumers[input_tensor].append(node)

            for output_tensor in node.output:
                self._producer[output_tensor] = node

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

        if tensor.data_type == mace_pb2.DT_INT32:
            tensor.int32_data.extend(value.astype(np.int32).flat)
        elif tensor.data_type == mace_pb2.DT_FLOAT:
            tensor.float_data.extend(value.astype(np.float32).flat)
        else:
            mace_check(False, "Not supported tensor type: %s" % name)

    def infer_shapes(self):
        graph_def = self._onnx_model.graph
        input_all = [node.name for node in graph_def.input]
        input_initializer = [node.name for node in graph_def.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        for input in graph_def.input:
            if input.name not in net_feed_input:
                continue

            mace_check(
                input.name in self._option.input_nodes,
                "input node info is invalide")
            mace_input_node = self._option.input_nodes[input.name]
            input_shape = mace_input_node.shape
            if mace_input_node.data_format == DataFormat.NHWC and len(
                    input_shape) == 4:
                input_shape = [
                    input_shape[0],
                    input_shape[3],
                    input_shape[1],
                    input_shape[2]]

            for i, dim in enumerate(input.type.tensor_type.shape.dim):
                dim.dim_value = input_shape[i]
        self._onnx_model = onnx.shape_inference.infer_shapes(self._onnx_model)

    def remove_tensor(self, name):
        tensors = self._mace_net_def.tensors
        for i in range(len(tensors)):
            if tensors[i].name == name:
                del tensors[i]
                break

    def run(self):
        self.infer_shapes()
        graph_def = self._onnx_model.graph
        self.extract_shape_info(graph_def)
        self.convert_tensors(graph_def)
        self.convert_ops(graph_def)
        return self._mace_net_def, self._converter_info

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
    def unsqueeze_shape(shape, axis):
        new_shape = [n for n in shape]
        for n in axis:
            new_shape.insert(n, 1)
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
                       "Mace does not support onnx op type %s(%s) yet"
                       % (node.name, node.op_type))
            self._op_converters[node.op_type](node)

    def replace_input(self, node, orig, replace):
        for idx in range(len(node.input)):
            if node.input[idx] == orig:
                node.input[idx] = replace

    def convert_tensors(self, graph_def):
        initializer = graph_def.initializer
        if initializer:
            for init in initializer:
                tensor = self._mace_net_def.tensors.add()
                tensor.name = init.name
                pytorch = FrameworkType.PYTORCH.name.lower()
                if self._source_framework == pytorch and \
                        init.name in self._consumers:
                    consumers = self._consumers[init.name]
                    for quant_node in consumers:
                        if quant_node.op_type == \
                                OnnxOpType.QuantizeLinear.name:
                            dequant_node = \
                                self._consumers[quant_node.output[0]][0]
                            mace_check(dequant_node.op_type ==
                                       OnnxOpType.DequantizeLinear.name,
                                       'consumer of QuantizeLinear must be DequantizeLinear')  # noqa
                            for node in self._onnx_model.graph.node:
                                self.replace_input(node, dequant_node.output[0], init.name)
                            self._skip_quant_dequants.add(quant_node.output[0])
                            self._skip_quant_dequants.add(
                                dequant_node.output[0])

                onnx_tensor = numpy_helper.to_array(init)
                tensor.dims.extend(list(init.dims))
                data_type = onnx_dtype(init.data_type)

                if data_type == np.float32 or data_type == np.float64:
                    tensor.data_type = mace_pb2.DT_FLOAT
                    tensor.float_data.extend(
                        onnx_tensor.astype(np.float32).flat)
                elif data_type == np.int64 or data_type == np.int32:
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
                output_shape = op.output_shape.add()
                if output in self._graph_shapes_dict:
                    shape_info = self._graph_shapes_dict[output]
                    output_shape.dims.extend(shape_info)
                else:
                    MaceLogger.warning(
                        "%s does not have output shape." % output)

        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        data_type_arg.i = self._option.data_type
        if op.input[0] in self._option.input_nodes:
            data_type_arg.i = self._option.input_nodes[op.input[0]].data_type
        elif op.input[0] in self._producer:
            for producer_op in self._mace_net_def.op:
                if self._producer[node.inputs[0]].name == producer_op.name:
                    producer_data_type = ConverterUtil.get_arg(
                        producer_op, MaceKeyword.mace_op_data_type_str)
                    if producer_data_type:
                        data_type_arg.i = producer_data_type.i

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
            elif node.op_type == OnnxOpType.Elu.name:
                alpha_value = 1.0
            elif node.op_type == OnnxOpType.HardSigmoid.name:
                alpha_value = 0.2
            else:
                alpha_value = 0
        alpha_arg = op.arg.add()
        if node.op_type == OnnxOpType.HardSigmoid.name:
            alpha_arg.name = MaceKeyword.mace_hardsigmoid_alpha_str
        else:
            alpha_arg.name = MaceKeyword.mace_activation_coefficient_str
        alpha_arg.f = alpha_value

        if node.op_type == OnnxOpType.HardSigmoid.name:
            if "beta" in node.attrs:
                beta_value = node.attrs["beta"]
            else:
                beta_value = 0.5
            beta_arg = op.arg.add()
            beta_arg.name = MaceKeyword.mace_hardsigmoid_beta_str
            beta_arg.f = beta_value

    def convert_quantize_linear(self, node):
        mace_check(self._source_framework ==
                   FrameworkType.PYTORCH.name.lower(),
                   'Only support QuantizeLinear exported by PyTorch, '
                   'this is exported by {}'.format(self._source_framework))
        scale = self._scale_zeros[node.inputs[1]]
        zero_point = self._scale_zeros[node.inputs[2]]
        if len(node.outputs) == 1 and \
                node.outputs[0] in self._skip_quant_dequants:
            tensor_name = node.inputs[0]
            symmetric = (zero_point.dtype == np.int8)
            if symmetric:
                mace_check(zero_point == 0,
                           "Zero point must be zero for int8 quantization")
                self._converter_info[InfoKey.qat_type][tensor_name] = \
                    QatType.SYMMETRIC.value
            else:
                self._converter_info[InfoKey.qat_type][tensor_name] = \
                    QatType.ASYMMETRIC.value
            self._consts[tensor_name].scale = scale
            if not symmetric:
                self._consts[tensor_name].zero_point = zero_point
            self._converter_info[InfoKey.has_qat].add(tensor_name)
            return
        op = self.convert_general_op(node)
        minval, maxval = quantize_util.scale_zero_to_min_max(scale, zero_point)
        if op.input[0] in self._option.input_nodes:
            self._option.input_nodes[op.input[0]].range = [minval, maxval]
            op.type = MaceOp.Identity.name
            del op.input[1:]
        else:
            op.type = 'FakeQuantWithMinMaxVars'
            min_arg = op.arg.add()
            min_arg.name = 'min'
            max_arg = op.arg.add()
            max_arg.name = 'max'
            min_arg.f = minval
            max_arg.f = maxval
            num_bits_arg = op.arg.add()
            num_bits_arg.name = 'num_bits'
            num_bits_arg.i = 8
            narrow_range_arg = op.arg.add()
            narrow_range_arg.name = 'narrow_range'
            narrow_range_arg.i = 0
            del op.input[1:]

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
            if dtype == np.float32 or dtype == np.float64:
                op.output_type.extend([self._option.data_type])
            elif dtype == np.int64 or dtype == np.int32:
                op.output_type.extend([mace_pb2.DT_INT32])
            elif dtype == np.bool_:
                op.output_type.extend([mace_pb2.DT_BOOL])
            else:
                mace_check(False, "data type %s not supported" % dtype)
        else:
            op.output_type.extend([self._option.data_type])

    def convert_concat(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Concat.name
        if self._isKaldi is False:
            mace_check('axis' in node.attrs,
                       'Concat op should have axis attribute.')
            axis_value = node.attrs['axis']
        else:
            axis_value = -1
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis_value

    def convert_constant(self, node):
        output_name = node.outputs[0]
        onnx_tensor = node.attrs['value']
        tensor_value = numpy_helper.to_array(onnx_tensor)
        data_type = onnx_dtype(onnx_tensor.data_type)
        consumers = self._consumers[output_name]
        pytorch = FrameworkType.PYTORCH.name.lower()
        quantize = OnnxOpType.QuantizeLinear.name
        dequantize = OnnxOpType.DequantizeLinear.name
        if self._source_framework == pytorch and \
                len(consumers) == 1 and \
                consumers[0].op_type in [quantize, dequantize]:
            mace_check(tensor_value.size == 1,
                       'scale/zero_point must have only 1 element, '
                       '{} element is found'.format(tensor_value.size))
            mace_check(data_type == np.float32 or data_type == np.uint8 or
                       data_type == np.int8,
                       'scale/zero_point should have type of float32, '
                       'uint8 or int8')
            if consumers[0].op_type == OnnxOpType.DequantizeLinear:
                return
            self._scale_zeros[output_name] = tensor_value
            return
        tensor = self._mace_net_def.tensors.add()
        tensor.name = output_name
        if onnx_tensor.dims:
            tensor.dims.extend(list(onnx_tensor.dims))
        else:
            tensor.dims.extend([1])

        if data_type == np.float32 or data_type == np.float64:
            tensor.data_type = mace_pb2.DT_FLOAT
            tensor.float_data.extend(
                tensor_value.astype(np.float32).flat)
        elif data_type == np.int32 or data_type == np.int64:
            tensor.data_type = mace_pb2.DT_INT32
            tensor.int32_data.extend(
                tensor_value.astype(np.int32).flat)
        else:
            mace_check(False,
                       "Not supported tensor type: %s" % data_type)
        self._consts[tensor.name] = tensor

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
            if node.inputs[1] in self._graph_shapes_dict:
                filter_shape = self._graph_shapes_dict[node.inputs[1]]
            else:
                filter_shape = self._consts[node.inputs[1]].dims
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

    def convert_deconv(self, node):
        op = self.convert_general_op(node)

        self.add_stride_pad_kernel_arg(node.attrs, op)

        if 'group' in node.attrs:
            group_val = node.attrs["group"]
        else:
            group_val = 1
        if group_val > 1:
            if node.inputs[1] in self._graph_shapes_dict:
                filter_shape = self._graph_shapes_dict[node.inputs[1]]
            else:
                filter_shape = self._consts[node.inputs[1]].dims
            mace_check(group_val == filter_shape[0] and filter_shape[1] == 1,
                       'MACE does not support group deconv yet')
            op.type = MaceOp.DepthwiseDeconv2d.name
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
        if node.op_type == OnnxOpType.DepthToSpace.name:
            op.type = MaceOp.DepthToSpace.name
        else:
            op.type = MaceOp.SpaceToDepth.name
        mace_check(('blocksize' in node.attrs),
                   "DepthToSpace/SpaceToDepth must have blocksize attribute.")
        block_size = node.attrs['blocksize']
        size_arg = op.arg.add()
        size_arg.name = MaceKeyword.mace_space_depth_block_size_str
        size_arg.i = block_size

        if 'mode' in node.attrs:
            if node.attrs['mode'] == 'CRD':
                mode_arg = op.arg.add()
                mode_arg.name = 'mode'
                mode_arg.s = bytes('CRD', 'utf-8')
            else:
                mode_arg = op.arg.add()
                mode_arg.name = 'mode'
                mode_arg.s = bytes('DCR', 'utf-8')

    def convert_dequantize_linear(self, node):
        mace_check(self._source_framework ==
                   FrameworkType.PYTORCH.name.lower(),
                   'Only support DequantizeLinear exported by PyTorch, '
                   'this is exported by {}'.format(self._source_framework))
        if len(node.outputs) == 1 and \
                node.outputs[0] in self._skip_quant_dequants:
            return
        op = self.convert_general_op(node)
        op.type = MaceOp.Identity.name
        del op.input[1:]

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

    def convert_dropout(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Identity.name
        del op.output[1:]
        del op.output_shape[1:]

    def convert_dynamic_lstm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.DynamicLSTM.name

        self.copy_node_attr(op, node, 'prev_out_delay',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'prev_cell_delay',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'prev_out_offset',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'prev_out_dim',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'prev_cell_dim',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'bias_a',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'bias_b',
                            AttributeType.INT)
        self.copy_node_attr(op, node, 'scale',
                            AttributeType.FLOAT)
        self.copy_node_attr(op, node, 'subsample_factor',
                            AttributeType.INT, default=1)
        self.copy_node_attr(op, node, 'cell_cache_indexes',
                            AttributeType.INTS, default=[])
        self.copy_node_attr(op, node, 'out_cache_indexes',
                            AttributeType.INTS, default=[])
        self.copy_node_attr(op, node, 'forward_indexes',
                            AttributeType.INTS)

    def convert_clip(self, node):
        #  If clip's min value is zero,
        #  convert clip to activation(ReLU or ReLUX)
        #  so it can be fused into convolution.
        is_relux = False
        inputs_num = len(node.inputs)
        if 'min' in node.attrs or inputs_num > 1:
            if inputs_num > 1:
                min_value = self._consts[node.inputs[1]].float_data[0]
            else:
                min_value = node.attrs['min']
            if min_value == 0:
                is_relux = True
        if is_relux:
            op = self.convert_general_op(node)
            op.type = MaceOp.Activation.name

            type_arg = op.arg.add()
            type_arg.name = MaceKeyword.mace_activation_type_str
            if "max" in node.attrs or inputs_num > 2:
                if inputs_num > 2:
                    max_value = self._consts[node.inputs[2]].float_data[0]
                else:
                    max_value = node.attrs["max"]
                type_arg.s = six.b(ActivationType.RELUX.name)
                alpha_arg = op.arg.add()
                alpha_arg.name = MaceKeyword.mace_activation_max_limit_str
                alpha_arg.f = max_value
            else:
                type_arg.s = six.b(ActivationType.RELU.name)
            for input_name in op.input[1:]:
                if input_name in self._consts:
                    const_tensor = self._consts[input_name]
                    if const_tensor in self._mace_net_def.tensors:
                        self._mace_net_def.tensors.remove(const_tensor)
            if inputs_num > 1:
                del op.input[1:]
        else:
            self.convert_eltwise(node)
            return

    def convert_eltwise(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Eltwise.name
        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        type_arg.i = self.eltwise_type[node.op_type].value
        inputs_num = len(node.inputs)
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
            if inputs_num > 1:
                min_value = self._consts[node.inputs[1]].float_data[0]
            elif 'min' in node.attrs:
                min_value = node.attrs['min']
            else:
                min_value = np.finfo(np.float32).min
            if inputs_num > 1:
                max_value = self._consts[node.inputs[2]].float_data[0]
            elif 'max' in node.attrs:
                max_value = node.attrs['max']
            else:
                max_value = np.finfo(np.float32).max
            coeff_arg = op.arg.add()
            coeff_arg.name = MaceKeyword.mace_coeff_str
            coeff_arg.floats.extend([min_value, max_value])
            for input_name in op.input[1:]:
                if input_name in self._consts:
                    const_tensor = self._consts[input_name]
                    self._mace_net_def.tensors.remove(const_tensor)
            del op.input[1:]
        elif len(node.inputs) == 2:
            if node.inputs[1] in self._consts and \
                    node.inputs[0] not in self._consts:
                const_name = node.inputs[1]
                const_tensor = self._consts[const_name]
                dims = const_tensor.dims
                if len(dims) == 0 or (len(dims) == 1 and dims[0] == 1):
                    value_arg = op.arg.add()
                    value_arg.name = MaceKeyword.mace_scalar_input_str
                    if const_tensor.data_type == mace_pb2.DT_INT32:
                        value_arg.f = float(const_tensor.int32_data[0])
                    elif const_tensor.data_type == mace_pb2.DT_FLOAT:
                        value_arg.f = const_tensor.float_data[0]
                    else:
                        mace_check(False,
                                   "Does not support param's data type %s"
                                   % const_tensor.data_type)
                    value_index_arg = op.arg.add()
                    value_index_arg.name = \
                        MaceKeyword.mace_scalar_input_index_str
                    value_index_arg.i = 1
                    del op.input[1]
            elif node.inputs[0] in self._consts and \
                    node.inputs[1] not in self._consts:
                const_name = node.inputs[0]
                const_tensor = self._consts[const_name]
                dims = const_tensor.dims
                if len(dims) == 0 or (len(dims) == 1 and dims[0] == 1):
                    value_arg = op.arg.add()
                    value_arg.name = MaceKeyword.mace_scalar_input_str
                    if const_tensor.data_type == mace_pb2.DT_INT32:
                        value_arg.f = float(const_tensor.int32_data[0])
                    elif const_tensor.data_type == mace_pb2.DT_FLOAT:
                        value_arg.f = const_tensor.float_data[0]
                    else:
                        mace_check(False,
                                   "Does not support param's data type %s"
                                   % const_tensor.data_type)
                    value_index_arg = op.arg.add()
                    value_index_arg.name = \
                        MaceKeyword.mace_scalar_input_index_str
                    value_index_arg.i = 0
                    del op.input[0]

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
        self.copy_node_attr(op, node, 'counts', AttributeType.FLOATS)
        self.copy_node_attr(op, node, 'forward_indexes', AttributeType.INTS)

    def convert_flatten(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reshape.name
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = 1
        if 'axis' in node.attrs:
            axis_arg.i = node.attrs['axis']
        axis_arg.i = 4 + axis_arg.i if axis_arg.i < 0 else axis_arg.i

        end_axis_arg = op.arg.add()
        end_axis_arg.name = MaceKeyword.mace_end_axis_str
        end_axis_arg.i = -1

    def convert_kaldi_batchnorm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.KaldiBatchNorm.name
        dim = self.copy_node_attr(op, node, 'dim', AttributeType.INT, -1)
        block_dim = self.copy_node_attr(op, node, 'block_dim',
                                        AttributeType.INT, -1)
        epsilon = self.copy_node_attr(op, node, 'epsilon',
                                      AttributeType.FLOAT, 1e-3)
        target_rms = self.copy_node_attr(op, node, 'target_rms',
                                         AttributeType.FLOAT, 1.0)
        test_mode = self.copy_node_attr(op, node, 'test_mode',
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
        scale_value = ((1.0 / np.sqrt(
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

    def convert_instance_norm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.InstanceNorm.name
        if "epsilon" in node.attrs:
            epsilon_value = node.attrs["epsilon"]
        else:
            epsilon_value = 1e-5
        epsilon_arg = op.arg.add()
        epsilon_arg.name = MaceKeyword.mace_epsilon_str
        epsilon_arg.f = epsilon_value
        mace_check(len(node.inputs) == 3, "instance norm must have 3 inputs.")
        scale_value = np.array(self._consts[node.inputs[1]].float_data)
        offset_value = np.array(self._consts[node.inputs[2]].float_data)
        scale_are_ones = np.array_equal(
            scale_value, np.ones(len(scale_value), dtype=np.float32))
        offset_are_zeros = np.array_equal(
            offset_value, np.zeros(len(offset_value), dtype=np.float32))
        affine = not (scale_are_ones and offset_are_zeros)
        affine_arg = op.arg.add()
        affine_arg.name = MaceKeyword.mace_affine_str
        affine_arg.i = int(affine)

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
        if op.input[1] in self._consts:
            indices = self._consts[op.input[1]]
            dims = indices.dims
            if len(dims) == 0 or (len(dims) == 1 and dims[0] == 1):
                op.output_shape[0].dims.insert(value, 1)

    def convert_gemm(self, node):
        if self._isKaldi:
            self.convert_affine(node)
            return

        mace_check(len(node.inputs) >= 2,
                   "Gemm should have at least two inputs.")
        if 'alpha' in node.attrs:
            alpha = node.attrs['alpha']
            if alpha != 1.0 and node.inputs[1] in self._consts:
                weights = self._consts[node.inputs[1]]
                for idx in six.moves.range(self.get_tensor_len(weights)):
                    weights.float_data[idx] *= alpha
        if 'beta' in node.attrs:
            beta = node.attrs['beta']
            if beta != 1.0 and len(node.inputs) == 3 and\
                    node.inputs[2] in self._consts:
                bias = self._consts[node.inputs[2]]
                for idx in six.moves.range(self.get_tensor_len(bias)):
                    bias.float_data[idx] *= beta
        trans_a = node.attrs['transA'] if 'transA' in node.attrs else 0
        trans_b = node.attrs['transB'] if 'transB' in node.attrs else 0
        is_fc = False
        if trans_a == 0 and trans_b == 1 and\
            node.inputs[0] in self._graph_shapes_dict and\
                node.inputs[1] in self._graph_shapes_dict and \
                node.inputs[1] in self._consts:
            shape_a = self._graph_shapes_dict[node.inputs[0]]
            shape_b = self._graph_shapes_dict[node.inputs[1]]
            if len(shape_a) == 4 and len(shape_b) == 2:
                tensor_b = self._consts[node.inputs[1]]
                tensor_data = np.array(tensor_b.float_data).reshape(
                    shape_b[0], shape_b[1], 1, 1)
                tensor_b.float_data[:] = tensor_data.flat
                tensor_b.dims[:] = tensor_data.shape
                is_fc = True
            elif len(shape_a) == 4 and\
                    len(shape_b) == 4 and list(shape_b[2:]) == [1, 1]:
                is_fc = True
        if is_fc:
            op = self.convert_general_op(node, with_shape=False)
            op.type = MaceOp.FullyConnected.name
            for output in node.outputs:
                output_shape = op.output_shape.add()
                shape_info = self._graph_shapes_dict[output]
                mace_check(len(shape_info) in [2, 4],
                           "gemm output shape should be 2 or 4 dims.")
                if len(shape_info) == 4:
                    mace_check(list(shape_info[2:]) == [1, 1],
                               "gemm's output shape should be [*, * , 1, 1]")
                else:
                    shape_info = [shape_info[0], shape_info[1], 1, 1]
                output_shape.dims.extend(shape_info)
        else:
            op = self.convert_general_op(node)
            op.type = MaceOp.MatMul.name
            trans_a_arg = op.arg.add()
            trans_a_arg.name = MaceKeyword.mace_transpose_a_str
            trans_a_arg.i = trans_a
            trans_b_arg = op.arg.add()
            trans_b_arg.name = MaceKeyword.mace_transpose_b_str
            trans_b_arg.i = trans_b

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
            op.type = MaceOp.IfDefined.name
            self.copy_node_attr(op, node, 'forward_indexes',
                                AttributeType.INTS)
            self.copy_node_attr(op, node, 'cache_forward_indexes',
                                AttributeType.INTS)

    def convert_imagescaler(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.BatchNorm.name

        scale = node.attrs['scale']
        bias_value = np.array(node.attrs['bias'])
        scale_value = scale * np.ones_like(bias_value)

        scale_name = node.name + "_scale"
        bias_name = node.name + "_bias"
        self.add_tensor(scale_name, scale_value.shape,
                        mace_pb2.DT_FLOAT, scale_value)
        self.add_tensor(bias_name, bias_value.shape,
                        mace_pb2.DT_FLOAT, bias_value)
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
                padding_type_arg.i = PadType.REFLECT.value
            elif mode == 'edge':
                padding_type_arg.i = PadType.SYMMETRIC.value
            else:
                padding_type_arg.i = PadType.CONSTANT.value
        if 'pads' in node.attrs:
            paddings_arg = op.arg.add()
            paddings_arg.name = MaceKeyword.mace_paddings_str
            paddings_value = node.attrs['pads']
            paddings_value = np.asarray(paddings_value).reshape(
                (2, -1)).transpose().reshape(-1).tolist()
            paddings_arg.ints.extend(paddings_value)
        if 'value' in node.attrs:
            constant_value_arg = op.arg.add()
            constant_value_arg.name = MaceKeyword.mace_constant_value_str
            constant_value_arg.f = node.attrs['value']
        if len(op.input) > 1:
            constant_value_arg = op.arg.add()
            constant_value_arg.name = MaceKeyword.mace_paddings_str
            paddings_value = self._consts[node.inputs[1]].int32_data
            paddings_value = np.asarray(paddings_value).reshape(
                (2, -1)).transpose().reshape(-1).tolist()
            constant_value_arg.ints.extend(paddings_value)
            del op.input[1:]

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
        if node.op_type == OnnxOpType.GlobalAveragePool.name:
            shape_info = self._graph_shapes_dict[node.inputs[0]]
            kernel_shape = shape_info[2:]
            kernel = [kernel_shape[0], kernel_shape[1]]
            kernels_arg = op.arg.add()
            kernels_arg.name = MaceKeyword.mace_kernel_str
            kernels_arg.ints.extend(kernel)

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

    def convert_replaceindex(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.ReplaceIndex.name
        self.copy_node_attr(op, node, 'forward_indexes',
                            AttributeType.INTS)

    def convert_reshape(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Reshape.name

    def convert_shape(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Shape.name
        op.output_type.extend([mace_pb2.DT_INT32])

    def convert_slice(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Slice.name

        if 'starts' in node.attrs:
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

        if not op.output_shape[0].dims:
            if node.inputs[0] in self._consts:
                tensor = self._consts[node.inputs[0]]
                if tensor.dims:
                    op.output_shape[0].dims.extend(tensor.dims)

    def convert_softmax(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Softmax.name
        if node.op_type == OnnxOpType.LogSoftmax.name:
            use_log_arg = op.arg.add()
            use_log_arg.name = 'use_log'
            use_log_arg.i = 1
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = node.attrs.get('axis', -1)

    def convert_lpnormalization(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.LpNorm.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = node.attrs.get('axis', -1)

        p_arg = op.arg.add()
        p_arg.name = MaceKeyword.mace_p_str
        p_arg.i = node.attrs.get('p', 2)

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
            const_dim_arg = op.arg.add()
            const_dim_arg.name = 'const_component_dim'
            const_dim_arg.i = const_dim
            self.copy_node_attr(op, node,
                                'forward_const_indexes',
                                AttributeType.INTS)

        self.copy_node_attr(op, node, 'subsample_factor',
                            AttributeType.INT, default=1)
        self.copy_node_attr(op, node, 'forward_indexes',
                            AttributeType.INTS)

    @staticmethod
    def is_equal_split(size_splits):
        equal_split = True
        for size in size_splits:
            if size != size_splits[0]:
                equal_split = False
        return equal_split

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

        if 'split' in node.attrs:
            size_splits = np.array(
                node.attrs['split'], dtype=np.int32).reshape(-1)
            if not self.is_equal_split(size_splits):
                tensor_name = op.input[0] + "_mace_node_size_split_input_"
                tensor_shape = [size_splits.size]
                data_type = mace_pb2.DT_INT32
                op.input.append(tensor_name)
                self.add_tensor(tensor_name, tensor_shape, data_type,
                                size_splits)
        elif len(node.inputs) > 1 and node.inputs[1] in self._consts and \
                self.is_equal_split(self._consts[node.inputs[1]].int32_data):
            del op.input[1:]

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
            if 'axes' in node.attrs:
                axis_value = node.attrs['axes']
            else:
                axis_value = []
            axis_arg.ints.extend(axis_value)

    def convert_unsqueeze(self, node):
        mace_check('axes' in node.attrs,
                   "Unsqueeze op should have 'axes' attribute.")
        axis_value = node.attrs['axes']
        if node.inputs[0] in self._consts:
            tensor = self._consts[node.inputs[0]]
            shape = tensor.dims
            new_shape = self.unsqueeze_shape(shape, axis_value)
            del tensor.dims[:]
            tensor.dims.extend(new_shape)
            self.remove_node(node)
        else:
            op = self.convert_general_op(node)
            op.type = MaceOp.Unsqueeze.name
            axis_arg = op.arg.add()
            axis_arg.name = MaceKeyword.mace_axis_str
            axis_arg.ints.extend(axis_value)

    def convert_subsample(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Subsample.name
        self.copy_node_attr(op, node, 'forward_indexes',
                            AttributeType.INTS)

    def convert_sum_group(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.SumGroup.name

    def convert_target_rms_norm(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.TargetRMSNorm.name

        self.copy_node_attr(op, node, 'target_rms',
                            AttributeType.FLOAT)
        self.copy_node_attr(op, node, 'add_log_stddev',
                            AttributeType.INT, default=0)
        self.copy_node_attr(op, node, 'block_dim',
                            AttributeType.INT, default=0)

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

    def convert_upsample(self, node):
        op = self.convert_general_op(node)

        if node.attrs['mode'] == 'nearest':
            op.type = MaceOp.ResizeNearestNeighbor.name
        else:
            op.type = MaceOp.ResizeBilinear.name

        scale_tensor = self._consts[node.inputs[1]]
        height_scale_arg = op.arg.add()
        height_scale_arg.name = MaceKeyword.mace_height_scale_str
        width_scale_arg = op.arg.add()
        width_scale_arg.name = MaceKeyword.mace_width_scale_str
        height_scale_arg.f = scale_tensor.float_data[2]
        width_scale_arg.f = scale_tensor.float_data[3]

        align_corners_arg = op.arg.add()
        align_corners_arg.name = MaceKeyword.mace_align_corners_str
        align_corners_arg.i = node.attrs.get('align_corners', 0)

    def convert_resize(self, node):
        op = self.convert_general_op(node)
        if len(op.input) >= 3:
            roi_tensor = self._consts[op.input[1]]
            mace_check(len(roi_tensor.dims) == 0 or roi_tensor.dims[0] == 0 or
                       (len(roi_tensor.dims) > 0 and node.attrs['mode'] != "tf_crop_and_resize"),
                       "Unsupport resize roi")

        del op.input[1:]

        if self._opset_version <= 10:
            scale_tensor = self._consts[node.inputs[1]]
        else:
            scale_tensor = self._consts[node.inputs[2]]
            if scale_tensor.dims[0] == 0:
                size_tensor = self._consts[node.inputs[3]]
                size_arg = op.arg.add()
                size_arg.name = MaceKeyword.mace_resize_size_str
                size_value = np.array([size_tensor.int32_data[-2],
                                       size_tensor.int32_data[-1]],
                                      dtype=np.int32)
                size_arg.ints.extend(size_value)
        if scale_tensor.dims[0] != 0:
            height_scale_arg = op.arg.add()
            height_scale_arg.name = MaceKeyword.mace_height_scale_str
            width_scale_arg = op.arg.add()
            width_scale_arg.name = MaceKeyword.mace_width_scale_str
            height_scale_arg.f = scale_tensor.float_data[-2]
            width_scale_arg.f = scale_tensor.float_data[-1]
        for name in op.input[1:]:
            if name in self._consts:
                self.remove_tensor(name)

        if node.attrs['mode'] == 'nearest':
            op.type = MaceOp.ResizeNearestNeighbor.name
        elif node.attrs['mode'] == 'linear':
            op.type = MaceOp.ResizeBilinear.name
        elif node.attrs['mode'] == 'cubic':
            op.type = MaceOp.ResizeBicubic.name
        else:
            mace_check(False, "Unsupported mode %s" % node.attrs['mode'])

        if self._opset_version >= 11:
            ct_mode = node.attrs['coordinate_transformation_mode']
            # Only support pytorch resize, i.e. 'asymmetric' for 'nearest' and
            # ['align_corners', 'pytorch_half_pixel'] for ['linear', 'cubic']
            if op.type == MaceOp.ResizeNearestNeighbor.name:
                nearest_mode_arg = op.arg.add()
                nearest_mode_arg.name = MaceKeyword.mace_nearest_mode_str
                if 'nearest_mode' in node.attrs:
                    nearest_mode_arg.s = six.b(node.attrs['nearest_mode'])
                else:
                    # ONNX model exported by paddle has no nearest_mode, ONNX's default is round_prefer_floor
                    nearest_mode_arg.s = six.b('round_prefer_floor')
                mace_check(nearest_mode_arg.s == six.b("round_prefer_floor") or
                           nearest_mode_arg.s == six.b("floor"),
                           "Only support round_prefer_floor or floor, but "
                           "{} is got".format(nearest_mode_arg.s))
                mace_check(ct_mode == 'asymmetric',
                           "Resize nearest doesn't support: %s" % ct_mode)
            elif op.type in [MaceOp.ResizeBilinear.name,
                             MaceOp.ResizeBicubic.name]:
                mace_check(ct_mode == 'align_corners' or
                           ct_mode == 'pytorch_half_pixel',
                           "Resize linear/cubic doesn't support: %s" % ct_mode)
            if ct_mode == 'align_corners':
                align_corners_arg = op.arg.add()
                align_corners_arg.name = MaceKeyword.mace_align_corners_str
                align_corners_arg.i = 1
            elif ct_mode == 'pytorch_half_pixel':
                coordinate_transformation_mode_arg = op.arg.add()
                coordinate_transformation_mode_arg.name = \
                    MaceKeyword.mace_coordinate_transformation_mode_str
                coordinate_transformation_mode_arg.i = \
                    CoordinateTransformationMode.PYTORCH_HALF_PIXEL.value
        del op.input[1:]

    def convert_where(self, node):
        op = self.convert_general_op(node)
        op.type = MaceOp.Where.name
        data_type = ConverterUtil.get_arg(op, MaceKeyword.mace_op_data_type_str)
        if node.inputs[1] in self._consts:
            data_type.i = self._consts[node.inputs[1]].data_type
        elif node.inputs[2] in self._consts:
            data_type.i = self._consts[node.inputs[2]].data_type
