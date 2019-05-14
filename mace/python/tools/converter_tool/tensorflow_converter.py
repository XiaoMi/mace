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

import os
import math
import numpy as np
import six
import tensorflow as tf
from enum import Enum

from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter
from mace.python.tools.converter_tool.base_converter import PoolingType
from mace.python.tools.converter_tool.base_converter import PaddingMode
from mace.python.tools.converter_tool.base_converter import ActivationType
from mace.python.tools.converter_tool.base_converter import EltwiseType
from mace.python.tools.converter_tool.base_converter import PadType
from mace.python.tools.converter_tool.base_converter import FrameworkType
from mace.python.tools.converter_tool.base_converter import ReduceType
from mace.python.tools.converter_tool.base_converter import DataFormat
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.convert_util import mace_check

from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.tools.graph_transforms import TransformGraph

tf_padding_str = 'padding'
tf_strides_str = 'strides'
tf_dilations_str = 'dilations'
tf_data_format_str = 'data_format'
tf_kernel_str = 'ksize'
tf_epsilon_str = 'epsilon'
tf_alpha_str = 'alpha'
tf_is_training_str = 'is_training'
tf_align_corners = 'align_corners'
tf_block_size = 'block_size'
tf_squeeze_dims = 'squeeze_dims'
tf_axis = 'axis'

TFSupportedOps = [
    'Conv2D',
    'DepthwiseConv2dNative',
    'Conv2DBackpropInput',
    'BiasAdd',
    'Add',
    'Sub',
    'Mul',
    'Div',
    'Min',
    'Minimum',
    'Max',
    'Maximum',
    'Neg',
    'Abs',
    'Pow',
    'RealDiv',
    'Square',
    'SquaredDifference',
    'Rsqrt',
    'Equal',
    'Relu',
    'LeakyRelu',
    'Relu6',
    'Tanh',
    'Sigmoid',
    'Fill',
    'FusedBatchNorm',
    'AvgPool',
    'MaxPool',
    'ExpandDims',
    'Squeeze',
    'MatMul',
    'BatchMatMul',
    'Identity',
    'Reshape',
    'Shape',
    'Transpose',
    'Softmax',
    'ResizeBicubic',
    'ResizeBilinear',
    'ResizeNearestNeighbor',
    'Placeholder',
    'SpaceToBatchND',
    'BatchToSpaceND',
    'DepthToSpace',
    'SpaceToDepth',
    'Pad',
    'ConcatV2',
    'Mean',
    'Prod',
    'Const',
    'Gather',
    'GatherV2',
    'StridedSlice',
    'Slice',
    'ReverseV2',
    'Stack',
    'Pack',
    'Unstack',
    'Unpack',
    'Cast',
    'ArgMax',
    'Split',
    'FakeQuantWithMinMaxVars',
    'FakeQuantWithMinMaxArgs',
    'FloorDiv',
    'Sqrt',
    'MirrorPad',
    'Cumsum',
    'OneHot',
]

TFOpType = Enum('TFOpType', [(op, op) for op in TFSupportedOps], type=str)

TFSupportedOps = [six.b(op) for op in TFSupportedOps]

TFTransformGraphOptions = [
    'strip_unused_nodes',
    'remove_nodes(op=Identity, op=CheckNumerics)',
    'fold_constants(ignore_errors=true)',
    'fold_batch_norms',
    'fold_old_batch_norms',
    'remove_control_dependencies',
    'strip_unused_nodes',
    'sort_by_execution_order'
]


class TensorflowConverter(base_converter.ConverterInterface):
    """A class for convert tensorflow frozen model to mace model.
    We use tensorflow engine to infer op output shapes, since they are of
    too many types."""

    padding_mode = {
        'VALID': PaddingMode.VALID,
        'SAME': PaddingMode.SAME,
        'FULL': PaddingMode.FULL
    }
    padding_mode = {six.b(k): v for k, v in six.iteritems(padding_mode)}

    pooling_type_mode = {
        TFOpType.AvgPool.name: PoolingType.AVG,
        TFOpType.MaxPool.name: PoolingType.MAX
    }

    eltwise_type = {
        TFOpType.Add.name: EltwiseType.SUM,
        TFOpType.Sub.name: EltwiseType.SUB,
        TFOpType.Mul.name: EltwiseType.PROD,
        TFOpType.Div.name: EltwiseType.DIV,
        TFOpType.Minimum.name: EltwiseType.MIN,
        TFOpType.Maximum.name: EltwiseType.MAX,
        TFOpType.Neg.name: EltwiseType.NEG,
        TFOpType.Abs.name: EltwiseType.ABS,
        TFOpType.Pow.name: EltwiseType.POW,
        TFOpType.RealDiv.name: EltwiseType.DIV,
        TFOpType.FloorDiv.name: EltwiseType.FLOOR_DIV,
        TFOpType.SquaredDifference.name: EltwiseType.SQR_DIFF,
        TFOpType.Square.name: EltwiseType.POW,
        TFOpType.Rsqrt.name: EltwiseType.POW,
        TFOpType.Sqrt.name: EltwiseType.POW,
        TFOpType.Equal.name: EltwiseType.EQUAL,
    }

    activation_type = {
        TFOpType.Relu.name: ActivationType.RELU,
        TFOpType.Relu6.name: ActivationType.RELUX,
        TFOpType.Tanh.name: ActivationType.TANH,
        TFOpType.Sigmoid.name: ActivationType.SIGMOID,
        TFOpType.LeakyRelu.name: ActivationType.LEAKYRELU,
    }

    reduce_math_type = {
        TFOpType.Min.name: ReduceType.MIN,
        TFOpType.Max.name: ReduceType.MAX,
        TFOpType.Mean.name: ReduceType.MEAN,
        TFOpType.Prod.name: ReduceType.PROD,
    }

    pad_type = {
        'CONSTANT':  PadType.CONSTANT,
        'REFLECT':   PadType.REFLECT,
        'SYMMETRIC': PadType.SYMMETRIC
    }

    def __init__(self, option, src_model_file):
        self._op_converters = {
            TFOpType.Conv2D.name: self.convert_conv2d,
            TFOpType.DepthwiseConv2dNative.name: self.convert_conv2d,
            TFOpType.Conv2DBackpropInput.name: self.convert_conv2d,
            TFOpType.BiasAdd.name: self.convert_biasadd,
            TFOpType.Add.name: self.convert_add,
            TFOpType.Sub.name: self.convert_elementwise,
            TFOpType.Mul.name: self.convert_elementwise,
            TFOpType.Div.name: self.convert_elementwise,
            TFOpType.Minimum.name: self.convert_elementwise,
            TFOpType.Maximum.name: self.convert_elementwise,
            TFOpType.Neg.name: self.convert_elementwise,
            TFOpType.Abs.name: self.convert_elementwise,
            TFOpType.Pow.name: self.convert_elementwise,
            TFOpType.RealDiv.name: self.convert_elementwise,
            TFOpType.SquaredDifference.name: self.convert_elementwise,
            TFOpType.Square.name: self.convert_elementwise,
            TFOpType.Rsqrt.name: self.convert_elementwise,
            TFOpType.Equal.name: self.convert_elementwise,
            TFOpType.Min.name: self.convert_reduce,
            TFOpType.Max.name: self.convert_reduce,
            TFOpType.Mean.name: self.convert_reduce,
            TFOpType.Prod.name: self.convert_reduce,
            TFOpType.Relu.name: self.convert_activation,
            TFOpType.LeakyRelu.name: self.convert_activation,
            TFOpType.Relu6.name: self.convert_activation,
            TFOpType.Tanh.name: self.convert_activation,
            TFOpType.Sigmoid.name: self.convert_activation,
            TFOpType.Fill.name: self.convert_fill,
            TFOpType.FusedBatchNorm.name: self.convert_fused_batchnorm,
            TFOpType.AvgPool.name: self.convert_pooling,
            TFOpType.MaxPool.name: self.convert_pooling,
            TFOpType.MatMul.name: self.convert_matmul,
            TFOpType.BatchMatMul.name: self.convert_matmul,
            TFOpType.Identity.name: self.convert_identity,
            TFOpType.Reshape.name: self.convert_reshape,
            TFOpType.Shape.name: self.convert_shape,
            TFOpType.ExpandDims.name: self.convert_expand_dims,
            TFOpType.Squeeze.name: self.convert_squeeze,
            TFOpType.Transpose.name: self.convert_transpose,
            TFOpType.Softmax.name: self.convert_softmax,
            TFOpType.ResizeBicubic.name: self.convert_resize_bicubic,
            TFOpType.ResizeBilinear.name: self.convert_resize_bilinear,
            TFOpType.ResizeNearestNeighbor.name: self.convert_resize_nearest_neighbor,  # noqa
            TFOpType.Placeholder.name: self.convert_nop,
            TFOpType.SpaceToBatchND.name: self.convert_space_batch,
            TFOpType.BatchToSpaceND.name: self.convert_space_batch,
            TFOpType.DepthToSpace.name: self.convert_space_depth,
            TFOpType.SpaceToDepth.name: self.convert_space_depth,
            TFOpType.Pad.name: self.convert_pad,
            TFOpType.ConcatV2.name: self.convert_concat,
            TFOpType.Const.name: self.convert_nop,
            TFOpType.Gather.name: self.convert_gather,
            TFOpType.GatherV2.name: self.convert_gather,
            TFOpType.StridedSlice.name: self.convert_stridedslice,
            TFOpType.Slice.name: self.convert_slice,
            TFOpType.ReverseV2.name: self.convert_reverse,
            TFOpType.Pack.name: self.convert_stack,
            TFOpType.Stack.name: self.convert_stack,
            TFOpType.Unpack.name: self.convert_unstack,
            TFOpType.Unstack.name: self.convert_unstack,
            TFOpType.Cast.name: self.convert_cast,
            TFOpType.ArgMax.name: self.convert_argmax,
            TFOpType.Split.name: self.convert_split,
            TFOpType.FakeQuantWithMinMaxVars.name: self.convert_fake_quantize,
            TFOpType.FakeQuantWithMinMaxArgs.name: self.convert_fake_quantize,
            TFOpType.FloorDiv.name: self.convert_elementwise,
            TFOpType.Sqrt.name: self.convert_elementwise,
            TFOpType.MirrorPad.name: self.convert_pad,
            TFOpType.Cumsum.name: self.convert_cumsum,
            TFOpType.OneHot.name: self.convert_one_hot,
        }
        self._option = option
        self._mace_net_def = mace_pb2.NetDef()
        ConverterUtil.set_filter_format(self._mace_net_def, DataFormat.HWIO)
        ConverterUtil.add_data_format_arg(self._mace_net_def, DataFormat.NHWC)

        # import tensorflow graph
        tf_graph_def = tf.GraphDef()
        with tf.gfile.Open(src_model_file, 'rb') as f:
            tf_graph_def.ParseFromString(f.read())

        self._placeholders = {}
        self._skip_tensor = set()
        self._output_shape = {}

        print("Run transform_graph: %s" % TFTransformGraphOptions)
        try:
            print("output keys: ", option.output_nodes.keys())
            transformed_graph_def = TransformGraph(tf_graph_def,
                                                   option.input_nodes.keys(),
                                                   option.output_nodes.keys(),
                                                   TFTransformGraphOptions)
        except Exception as ex:
            print("Failed to transform graph using tf tool: %s" % ex)
            transformed_graph_def = tf_graph_def

        # To check optimized model, uncomment following code.
        # tf.io.write_graph(
        #     transformed_graph_def,
        #     ".",
        #     os.path.basename(src_model_file)[:-3] + "_opt.pb",
        #     as_text=False
        # )

        self.add_shape_info(transformed_graph_def)

        with tf.Session() as session:
            with session.graph.as_default() as graph:
                tf.import_graph_def(transformed_graph_def, name='')
                self._tf_graph = graph
                self.update_output_shapes(session)

        # we have polluted graph with 'shape' ops, so reset it and reload it
        # again
        tf.reset_default_graph()

        with tf.Session() as session:
            with session.graph.as_default() as graph:
                tf.import_graph_def(transformed_graph_def, name='')
                self._tf_graph = graph

    def run(self):
        with tf.Session() as session:
            self.convert_ops(session)

        self.replace_input_output_tensor_name()
        return self._mace_net_def

    def replace_input_output_tensor_name(self):
        for op in self._mace_net_def.op:
            for i in six.moves.range(len(op.input)):
                if op.input[i][-2:] == ':0':
                    op_name = op.input[i][:-2]
                    if op_name in self._option.input_nodes \
                            or op_name in self._option.output_nodes:
                        op.input[i] = op_name
            for i in six.moves.range(len(op.output)):
                if op.output[i][-2:] == ':0':
                    op_name = op.output[i][:-2]
                    if op_name in self._option.output_nodes:
                        op.output[i] = op_name

    def add_shape_info(self, tf_graph_def):
        for node in tf_graph_def.node:
            for input_node in self._option.input_nodes.values():
                if node.name == input_node.name \
                        or node.name + ':0' == input_node.name:
                    input_shape = input_node.shape
                    if input_node.data_format == DataFormat.OIHW \
                            and len(input_shape) == 4:
                        # OIHW -> HWIO
                        input_shape = [input_shape[2], input_shape[3],
                                       input_shape[1], input_shape[0]]
                    del node.attr['shape'].shape.dim[:]
                    node.attr['shape'].shape.dim.extend([
                        tensor_shape_pb2.TensorShapeProto.Dim(size=i) for i in
                        input_shape
                    ])
                    self._placeholders[node.name + ':0'] = \
                        np.zeros(shape=input_shape, dtype=float)

    @staticmethod
    def get_scope(tensor_name):
        idx = tensor_name.rfind('/')
        if idx == -1:
            return tensor_name
        else:
            return tensor_name[:idx]

    def update_output_shapes(self, sess):
        tensors = []
        shape_tensors = []
        for tf_op in self._tf_graph.get_operations():
            for output in tf_op.outputs:
                tensors.append(output.name)
                shape_tensors.append(tf.shape(output))

        tensor_shapes = sess.run(shape_tensors,
                                 feed_dict=self._placeholders)
        for i in range(len(tensors)):
            self._output_shape[tensors[i]] = tensor_shapes[i]

    def convert_ops(self, sess):
        for tf_op in self._tf_graph.get_operations():
            mace_check(tf_op.type in self._op_converters,
                       "Mace does not support tensorflow op type %s yet"
                       % tf_op.type)
            self._op_converters[tf_op.type](tf_op)

        self.convert_tensors()

    def convert_tensors(self):
        for tf_op in self._tf_graph.get_operations():
            if tf_op.type != TFOpType.Const.name:
                continue
            output_name = tf_op.outputs[0].name
            if output_name not in self._skip_tensor:
                tensor = self._mace_net_def.tensors.add()
                tensor.name = tf_op.outputs[0].name
                tf_tensor = tf_op.outputs[0].eval()
                tensor.dims.extend(list(tf_tensor.shape))

                tf_dt = tf_op.get_attr('dtype')
                if tf_dt == tf.float32:
                    tensor.data_type = mace_pb2.DT_FLOAT
                    tensor.float_data.extend(tf_tensor.astype(np.float32).flat)
                elif tf_dt == tf.int32:
                    tensor.data_type = mace_pb2.DT_INT32
                    tensor.int32_data.extend(tf_tensor.astype(np.int32).flat)
                else:
                    mace_check(False,
                               "Not supported tensor type: %s" % tf_dt.name)

    def add_tensor(self, name, shape, data_type, value):
        tensor = self._mace_net_def.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_type = data_type
        tensor.float_data.extend(value.flat)

    # this function tries to infer tensor shape, but some dimension shape
    # may be undefined due to variance of input length
    def infer_tensor_shape(self, tensor, output_shape=None):
        shape = None
        if tensor.name in self._output_shape:
            shape = self._output_shape[tensor.name]
        else:
            shape = tensor.shape.as_list()

        if output_shape:
            output_shape.dims.extend(shape)

        return shape

    def convert_nop(self, tf_op):
        pass

    def convert_general_op(self, tf_op):
        op = self._mace_net_def.op.add()
        op.name = tf_op.name
        op.type = tf_op.type
        op.input.extend([tf_input.name for tf_input in tf_op.inputs])
        op.output.extend([tf_output.name for tf_output in tf_op.outputs])
        for tf_output in tf_op.outputs:
            output_shape = op.output_shape.add()
            self.infer_tensor_shape(tf_output, output_shape)

        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        try:
            dtype = tf_op.get_attr('T')
            if dtype == tf.int32:
                data_type_arg.i = mace_pb2.DT_INT32
            elif dtype == tf.float32:
                data_type_arg.i = self._option.data_type
            else:
                mace_check(False, "data type %s not supported" % dtype)
        except ValueError:
            try:
                dtype = tf_op.get_attr('SrcT')
                if dtype == tf.int32 or dtype == tf.bool:
                    data_type_arg.i = mace_pb2.DT_INT32
                elif dtype == tf.float32:
                    data_type_arg.i = self._option.data_type
                else:
                    mace_check(False, "data type %s not supported" % dtype)
            except ValueError:
                data_type_arg.i = self._option.data_type

        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.TENSORFLOW.value

        ConverterUtil.add_data_format_arg(op, DataFormat.NHWC)

        return op

    def convert_identity(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = 'Identity'

    def convert_conv2d(self, tf_op):
        op = self.convert_general_op(tf_op)
        if tf_op.type == TFOpType.DepthwiseConv2dNative.name:
            op.type = MaceOp.DepthwiseConv2d.name
        elif tf_op.type == TFOpType.Conv2DBackpropInput.name:
            op.type = MaceOp.Deconv2D.name
        else:
            op.type = MaceOp.Conv2D.name

        padding_arg = op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_str
        padding_arg.i = self.padding_mode[tf_op.get_attr(tf_padding_str)].value
        strides_arg = op.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(tf_op.get_attr(tf_strides_str)[1:3])
        if op.type != MaceOp.Deconv2D.name:
            dilation_arg = op.arg.add()
            dilation_arg.name = MaceKeyword.mace_dilations_str
            try:
                dilation_val = tf_op.get_attr(tf_dilations_str)[1:3]
            except ValueError:
                dilation_val = [1, 1]
            dilation_arg.ints.extend(dilation_val)
        else:
            try:
                dilation_val = tf_op.get_attr(tf_dilations_str)[1:3]
            except ValueError:
                dilation_val = [1, 1]
            mace_check(dilation_val[0] == 1 and dilation_val[1] == 1,
                       "Mace only supports dilation == 1 conv2d_transpose.")
            mace_check(len(tf_op.inputs) >= 3,
                       "deconv should have (>=) 3 inputs.")
            del op.input[:]
            op.input.extend([tf_op.inputs[2].name,
                             tf_op.inputs[1].name,
                             tf_op.inputs[0].name])

    def convert_elementwise(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Eltwise.name

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        type_arg.i = self.eltwise_type[tf_op.type].value

        def check_is_scalar(tf_op):
            if len(tf_op.inputs) == 1:
                return len(self.infer_tensor_shape(tf_op.inputs[0])) == 0
            elif len(tf_op.inputs) == 2:
                return len(self.infer_tensor_shape(tf_op.inputs[0])) == 0 and \
                       len(self.infer_tensor_shape(tf_op.inputs[1])) == 0

        if check_is_scalar(tf_op):
            op.type = MaceOp.ScalarMath.name
        else:
            op.type = MaceOp.Eltwise.name
        if tf_op.type == TFOpType.Square:
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = 2.0
        elif tf_op.type == TFOpType.Rsqrt:
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = -0.5
        elif tf_op.type == TFOpType.Sqrt:
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = 0.5

        if type_arg.i != EltwiseType.NEG.value \
                and type_arg.i != EltwiseType.ABS.value:
            try:
                def is_commutative(eltwise_type):
                    return EltwiseType(eltwise_type) in [
                        EltwiseType.SUM, EltwiseType.PROD,
                        EltwiseType.MAX, EltwiseType.MIN]

                if (len(tf_op.inputs) > 1 and
                        len(self.infer_tensor_shape(tf_op.inputs[1])) == 0 and
                        tf_op.inputs[1].op.type == TFOpType.Const.name):
                    scalar = tf_op.inputs[1].eval().astype(np.float32)
                    value_arg = op.arg.add()
                    value_arg.name = MaceKeyword.mace_scalar_input_str
                    value_arg.f = scalar
                    self._skip_tensor.add(tf_op.inputs[1].name)
                    value_index_arg = op.arg.add()
                    value_index_arg.name = \
                        MaceKeyword.mace_scalar_input_index_str
                    value_index_arg.i = 1
                    self._skip_tensor.add(tf_op.inputs[1].name)
                    del op.input[1]
                elif len(self.infer_tensor_shape(tf_op.inputs[0])) == 0 and \
                        tf_op.inputs[0].op.type == TFOpType.Const.name and \
                        is_commutative(type_arg.i):
                    scalar = tf_op.inputs[0].eval().astype(np.float32)
                    value_arg = op.arg.add()
                    value_arg.name = MaceKeyword.mace_scalar_input_str
                    value_arg.f = scalar
                    value_index_arg = op.arg.add()
                    value_index_arg.name = \
                        MaceKeyword.mace_scalar_input_index_str
                    value_index_arg.i = 0
                    self._skip_tensor.add(tf_op.inputs[0].name)
                    del op.input[0]
            except tf.errors.InvalidArgumentError:
                pass

    def convert_biasadd(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.BiasAdd.name

    def convert_one_hot(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.OneHot.name

        depth_arg = op.arg.add()
        depth_arg.name = 'depth'
        depth_arg.i = tf_op.inputs[1].eval().astype(np.int32)

        on_value_arg = op.arg.add()
        on_value_arg.name = 'on_value'
        on_value_arg.f = tf_op.inputs[2].eval().astype(np.float32)

        off_value_arg = op.arg.add()
        off_value_arg.name = 'off_value'
        off_value_arg.f = tf_op.inputs[3].eval().astype(np.float32)

        axis_arg = op.arg.add()
        axis_arg.name = tf_axis
        axis_arg.i = tf_op.get_attr(tf_axis)

        self._skip_tensor.update([inp.name for inp in tf_op.inputs][1:])
        del op.input[1:]

    def convert_add(self, tf_op):
        if len(tf_op.inputs) == 2:
            self.convert_elementwise(tf_op)
        else:
            op = self.convert_general_op(tf_op)
            op.type = MaceOp.AddN.name

    def convert_activation(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Activation.name

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_activation_type_str
        type_arg.s = six.b(self.activation_type[tf_op.type].name)

        if tf_op.type == TFOpType.Relu6.name:
            limit_arg = op.arg.add()
            limit_arg.name = MaceKeyword.mace_activation_max_limit_str
            limit_arg.f = 6.0
        elif tf_op.type == TFOpType.LeakyRelu.name:
            alpha_arg = op.arg.add()
            alpha_arg.name = \
                MaceKeyword.mace_activation_leakyrelu_coefficient_str
            alpha_arg.f = tf_op.get_attr(tf_alpha_str)

    def convert_fill(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Fill.name

    def convert_fused_batchnorm(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.BatchNorm.name

        is_training = tf_op.get_attr(tf_is_training_str)
        assert is_training is False, 'Only support batch normalization ' \
                                     'with is_training False, but got %s' % \
                                     is_training

        gamma_value = tf_op.inputs[1].eval().astype(np.float32)
        beta_value = tf_op.inputs[2].eval().astype(np.float32)
        mean_value = tf_op.inputs[3].eval().astype(np.float32)
        var_value = tf_op.inputs[4].eval().astype(np.float32)
        epsilon_value = tf_op.get_attr(tf_epsilon_str)

        scale_name = self.get_scope(tf_op.name) + '/scale:0'
        offset_name = self.get_scope(tf_op.name) + '/offset:0'
        scale_value = (
                (1.0 / np.vectorize(math.sqrt)(
                    var_value + epsilon_value)) * gamma_value)
        offset_value = (-mean_value * scale_value) + beta_value
        self.add_tensor(scale_name, scale_value.shape, mace_pb2.DT_FLOAT,
                        scale_value)
        self.add_tensor(offset_name, offset_value.shape, mace_pb2.DT_FLOAT,
                        offset_value)
        self._skip_tensor.update([inp.name for inp in tf_op.inputs][1:])

        del op.input[1:]
        op.input.extend([scale_name, offset_name])
        del op.output[1:]
        del op.output_shape[1:]

    def convert_pooling(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Pooling.name
        pooling_type_arg = op.arg.add()
        pooling_type_arg.name = MaceKeyword.mace_pooling_type_str
        pooling_type_arg.i = self.pooling_type_mode[tf_op.type].value
        padding_arg = op.arg.add()
        padding_arg.name = MaceKeyword.mace_padding_str
        padding_arg.i = self.padding_mode[tf_op.get_attr(tf_padding_str)].value
        strides_arg = op.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(tf_op.get_attr(tf_strides_str)[1:3])
        kernels_arg = op.arg.add()
        kernels_arg.name = MaceKeyword.mace_kernel_str
        kernels_arg.ints.extend(tf_op.get_attr(tf_kernel_str)[1:3])

    def convert_softmax(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Softmax.name

    def convert_resize_bicubic(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.ResizeBicubic.name
        del op.input[1:]

        size_arg = op.arg.add()
        size_arg.name = MaceKeyword.mace_resize_size_str
        size_value = tf_op.inputs[1].eval().astype(np.int32)
        size_arg.ints.extend(size_value)
        self._skip_tensor.add(tf_op.inputs[1].name)
        align_corners_arg = op.arg.add()
        align_corners_arg.name = MaceKeyword.mace_align_corners_str
        align_corners_arg.i = tf_op.get_attr(tf_align_corners)

    def convert_resize_bilinear(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.ResizeBilinear.name
        del op.input[1:]

        size_arg = op.arg.add()
        size_arg.name = MaceKeyword.mace_resize_size_str
        size_value = tf_op.inputs[1].eval().astype(np.int32)
        size_arg.ints.extend(size_value)
        self._skip_tensor.add(tf_op.inputs[1].name)
        align_corners_arg = op.arg.add()
        align_corners_arg.name = MaceKeyword.mace_align_corners_str
        align_corners_arg.i = tf_op.get_attr(tf_align_corners)

    def convert_resize_nearest_neighbor(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.ResizeNearestNeighbor.name

        align_corners_arg = op.arg.add()
        align_corners_arg.name = MaceKeyword.mace_align_corners_str
        align_corners_arg.i = tf_op.get_attr(tf_align_corners)

    def convert_space_batch(self, tf_op):
        op = self.convert_general_op(tf_op)
        del op.input[1:]

        size_arg = op.arg.add()
        size_arg.name = MaceKeyword.mace_space_batch_block_shape_str
        size_value = tf_op.inputs[1].eval().astype(np.int32)
        size_arg.ints.extend(size_value)

        crops_or_paddings_arg = op.arg.add()
        if op.type == TFOpType.BatchToSpaceND.name:
            op.type = MaceOp.BatchToSpaceND.name
            crops_or_paddings_arg.name = \
                MaceKeyword.mace_batch_to_space_crops_str
        else:
            op.type = MaceOp.SpaceToBatchND.name
            crops_or_paddings_arg.name = MaceKeyword.mace_paddings_str
        crops_or_paddings_value = tf_op.inputs[2].eval().astype(np.int32).flat
        crops_or_paddings_arg.ints.extend(crops_or_paddings_value)

        self._skip_tensor.add(tf_op.inputs[1].name)
        self._skip_tensor.add(tf_op.inputs[2].name)

    def convert_space_depth(self, tf_op):
        op = self.convert_general_op(tf_op)
        if op.type == TFOpType.SpaceToDepth.name:
            op.type = MaceOp.SpaceToDepth.name
        else:
            op.type = MaceOp.DepthToSpace.name

        size_arg = op.arg.add()
        size_arg.name = MaceKeyword.mace_space_depth_block_size_str
        size_arg.i = tf_op.get_attr(tf_block_size)

    def convert_pad(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Pad.name
        del op.input[1:]

        paddings_arg = op.arg.add()
        paddings_arg.name = MaceKeyword.mace_paddings_str
        paddings_value = tf_op.inputs[1].eval().astype(np.int32).flat
        paddings_arg.ints.extend(paddings_value)
        self._skip_tensor.add(tf_op.inputs[1].name)

        pad_type_arg = op.arg.add()
        pad_type_arg.name = MaceKeyword.mace_pad_type_str

        if tf_op.type == TFOpType.Pad:
            if len(tf_op.inputs) == 3:
                constant_value_arg = op.arg.add()
                constant_value_arg.name = MaceKeyword.mace_constant_value_str
                constant_value = tf_op.inputs[2].eval().astype(np.int32) \
                    .flat[0]
                constant_value_arg.i = constant_value
                self._skip_tensor.add(tf_op.inputs[2].name)

            pad_type_arg.i = PadType.CONSTANT.value

        elif tf_op.type == TFOpType.MirrorPad:
            pad_type_arg.i = self.pad_type[tf_op.get_attr('mode')].value

    def convert_concat(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Concat.name
        del op.input[-1]

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis = tf_op.inputs[-1].eval().astype(np.int32)
        axis = len(op.output_shape[0].dims) + axis if axis < 0 else axis
        axis_arg.i = axis

        self._skip_tensor.add(tf_op.inputs[-1].name)

    def convert_matmul(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.MatMul.name

        try:
            adj_x = tf_op.get_attr('adj_x')
            transpose_a_arg = op.arg.add()
            transpose_a_arg.name = MaceKeyword.mace_transpose_a_str
            transpose_a_arg.i = int(adj_x)
        except ValueError:
            try:
                transpose_a = tf_op.get_attr('transpose_a')
                transpose_a_arg = op.arg.add()
                transpose_a_arg.name = MaceKeyword.mace_transpose_a_str
                transpose_a_arg.i = int(transpose_a)
            except ValueError:
                pass

        try:
            adj_y = tf_op.get_attr('adj_y')
            transpose_b_arg = op.arg.add()
            transpose_b_arg.name = MaceKeyword.mace_transpose_b_str
            transpose_b_arg.i = int(adj_y)
        except ValueError:
            try:
                transpose_b = tf_op.get_attr('transpose_b')
                transpose_b_arg = op.arg.add()
                transpose_b_arg.name = MaceKeyword.mace_transpose_b_str
                transpose_b_arg.i = int(transpose_b)
            except ValueError:
                pass

    def convert_shape(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Shape.name
        op.output_type.extend([mace_pb2.DT_INT32])

    def convert_reshape(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Reshape.name

    def convert_expand_dims(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.ExpandDims.name

        axis_value = tf_op.inputs[1].eval().astype(np.int32)
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis_value
        del op.input[1]

    def convert_squeeze(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Squeeze.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        try:
            axis_value = tf_op.get_attr('squeeze_dims')
        except ValueError:
            try:
                axis_value = tf_op.get_attr('axis')
            except ValueError:
                axis_value = []
        axis_arg.ints.extend(axis_value)

    def convert_transpose(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Transpose.name

        perm = tf_op.inputs[1].eval().astype(np.int32)
        ordered_perm = np.sort(perm)

        if np.array_equal(perm, ordered_perm):
            op.type = MaceOp.Identity.name
            del op.input[1:]
            self._skip_tensor.add(tf_op.inputs[1].name)
        else:
            dims_arg = op.arg.add()
            dims_arg.name = MaceKeyword.mace_dims_str
            dims_arg.ints.extend(perm)

    def convert_reduce(self, tf_op):
        op = self.convert_general_op(tf_op)
        del op.input[1:]

        op.type = MaceOp.Reduce.name

        reduce_type_arg = op.arg.add()
        reduce_type_arg.name = MaceKeyword.mace_reduce_type_str
        reduce_type_arg.i = self.reduce_math_type[tf_op.type].value

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        if len(tf_op.inputs) > 1:
            reduce_dims = tf_op.inputs[1].eval()
        else:
            try:
                reduce_dims = tf_op.get_attr('axis')
            except ValueError:
                try:
                    reduce_dims = tf_op.get_attr('reduction_indices')
                except ValueError:
                    reduce_dims = []
        axis_arg.ints.extend(reduce_dims)
        keep_dims_arg = op.arg.add()
        keep_dims_arg.name = MaceKeyword.mace_keepdims_str
        try:
            keep_dims = tf_op.get_attr('keepdims')
        except ValueError:
            try:
                keep_dims = tf_op.get_attr('keep_dims')
            except ValueError:
                keep_dims = 0
        keep_dims_arg.i = keep_dims

        self._skip_tensor.add(tf_op.inputs[1].name)

    def convert_gather(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Gather.name

        if len(tf_op.inputs) >= 3:
            axis_arg = op.arg.add()
            axis_arg.name = MaceKeyword.mace_axis_str
            axis_arg.i = tf_op.inputs[2].eval()

    def convert_stridedslice(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.StridedSlice.name

        begin_mask_arg = op.arg.add()
        begin_mask_arg.name = MaceKeyword.mace_begin_mask_str
        begin_mask_arg.i = tf_op.get_attr(MaceKeyword.mace_begin_mask_str)

        end_mask_arg = op.arg.add()
        end_mask_arg.name = MaceKeyword.mace_end_mask_str
        end_mask_arg.i = tf_op.get_attr(MaceKeyword.mace_end_mask_str)

        ellipsis_mask_arg = op.arg.add()
        ellipsis_mask_arg.name = MaceKeyword.mace_ellipsis_mask_str
        ellipsis_mask_arg.i = tf_op.get_attr(
            MaceKeyword.mace_ellipsis_mask_str)

        new_axis_mask_arg = op.arg.add()
        new_axis_mask_arg.name = MaceKeyword.mace_new_axis_mask_str
        new_axis_mask_arg.i = tf_op.get_attr(
            MaceKeyword.mace_new_axis_mask_str)

        shrink_axis_mask_arg = op.arg.add()
        shrink_axis_mask_arg.name = MaceKeyword.mace_shrink_axis_mask_str
        shrink_axis_mask_arg.i = tf_op.get_attr(
            MaceKeyword.mace_shrink_axis_mask_str)

    def convert_slice(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.StridedSlice.name
        arg = op.arg.add()
        arg.name = 'slice'
        arg.i = 1

    def convert_reverse(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Reverse.name

    def convert_stack(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Stack.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        try:
            axis_arg.i = tf_op.get_attr(MaceKeyword.mace_axis_str)
        except ValueError:
            axis_arg.i = 0

    def convert_unstack(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Unstack.name

        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        try:
            axis_arg.i = tf_op.get_attr(MaceKeyword.mace_axis_str)
        except ValueError:
            axis_arg.i = 0

    def convert_cast(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Cast.name

        try:
            dtype = tf_op.get_attr('DstT')
            if dtype == tf.int32:
                op.output_type.extend([mace_pb2.DT_INT32])
            elif dtype == tf.float32:
                op.output_type.extend([self._option.data_type])
            else:
                mace_check(False, "data type %s not supported" % dtype)
        except ValueError:
            op.output_type.extend([self._option.data_type])

    def convert_argmax(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.ArgMax.name
        op.output_type.extend([mace_pb2.DT_INT32])

    def convert_split(self, tf_op):
        op = self.convert_general_op(tf_op)
        num_or_size_splits = tf_op.get_attr('num_split')
        if num_or_size_splits == 1:
            op.type = MaceOp.Identity.name
        else:
            op.type = MaceOp.Split.name
            axis = tf_op.inputs[0].eval().astype(np.int32)
            axis = len(op.output_shape[0].dims) + axis if axis < 0 else axis

            axis_arg = op.arg.add()
            axis_arg.name = MaceKeyword.mace_axis_str
            axis_arg.i = axis

            num_split_arg = op.arg.add()
            num_split_arg.name = MaceKeyword.mace_num_split_str
            num_split_arg.i = num_or_size_splits
        del op.input[0]
        self._skip_tensor.add(tf_op.inputs[0].name)

    def convert_fake_quantize(self, tf_op):
        op = self.convert_general_op(tf_op)
        min_arg = op.arg.add()
        min_arg.name = 'min'
        max_arg = op.arg.add()
        max_arg.name = 'max'
        if tf_op.type == TFOpType.FakeQuantWithMinMaxVars.name:
            min_arg.f = tf_op.inputs[1].eval()
            max_arg.f = tf_op.inputs[2].eval()
        elif tf_op.type == TFOpType.FakeQuantWithMinMaxArgs.name:
            min_arg.f = float(tf_op.get_attr('min'))
            max_arg.f = float(tf_op.get_attr('max'))
        narrow_range_arg = op.arg.add()
        narrow_range_arg.name = 'narrow_range'
        narrow_range_arg.i = int(tf_op.get_attr('narrow_range'))
        num_bits_arg = op.arg.add()
        num_bits_arg.name = 'num_bits'
        num_bits_arg.i = int(tf_op.get_attr('num_bits'))

        if tf_op.type == TFOpType.FakeQuantWithMinMaxVars.name:
            self._skip_tensor.add(tf_op.inputs[1].name)
            self._skip_tensor.add(tf_op.inputs[2].name)

    def convert_cumsum(self, tf_op):
        op = self.convert_general_op(tf_op)
        op.type = MaceOp.Cumsum.name

        axis = tf_op.inputs[1].eval().astype(np.int32)
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis
        del op.input[1]

        exclusive = tf_op.get_attr('exclusive')
        exclusive_arg = op.arg.add()
        exclusive_arg.name = MaceKeyword.mace_exclusive_str
        exclusive_arg.i = int(exclusive)

        reverse = tf_op.get_attr('reverse')
        reverse_arg = op.arg.add()
        reverse_arg.name = MaceKeyword.mace_reverse_str
        reverse_arg.i = int(reverse)
