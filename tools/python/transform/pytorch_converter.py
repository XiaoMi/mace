# Copyright 2020 The MACE Authors. All Rights Reserved.
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

import math
import torch
import sys
import re
import numpy as np
import six

from torch._C import _propagate_and_assign_input_shapes
# extract node attribute such as node['value']
from torch.onnx.utils import _node_getitem

from py_proto import mace_pb2
from transform import base_converter
from transform.transformer import Transformer
from transform.base_converter import ActivationType
from transform.base_converter import EltwiseType
from transform.base_converter import FrameworkType
from transform.base_converter import ConverterUtil
from transform.base_converter import DataFormat
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp
from transform.base_converter import PoolingType
from transform.base_converter import RoundMode
from utils.util import mace_check


def _model_to_graph(model, args):
    propagate = False
    if isinstance(args, torch.Tensor):
        args = (args, )

    graph = model.forward.graph
    method_graph, params = torch._C._jit_pass_lower_graph(graph, model._c)
    in_vars, in_desc = torch.jit._flatten(tuple(args) + tuple(params))
    graph = _propagate_and_assign_input_shapes(
        method_graph, tuple(in_vars), False, propagate)
    input_and_param_names = [val.debugName() for val in graph.inputs()]
    param_names = input_and_param_names[-len(params):]
    params = [elem.detach() for elem in params]
    params_dict = dict(zip(param_names, params))
    return graph, params_dict


class ValueType(object):
    BoolType = torch._C.BoolType
    FloatType = torch._C.FloatType
    IntType = torch._C.IntType
    ListType = torch._C.ListType
    NoneType = torch._C.NoneType
    TensorType = torch._C.TensorType
    TupleType = torch._C.TupleType


class NodeKind(object):
    AdaptiveAvgPool2D = 'aten::adaptive_avg_pool2d'
    Add_ = 'aten::add_'
    Add = 'aten::add'
    Addmm = 'aten::addmm'
    AvgPool2D = 'aten::avg_pool2d'
    BatchNorm = 'aten::batch_norm'
    Cat = 'aten::cat'
    Constant = 'prim::Constant'
    Convolution = 'aten::_convolution'
    Dropout = 'aten::dropout'
    Flatten = 'aten::flatten'
    HardTanh_ = 'aten::hardtanh_'  # ReLU6(inplace=True)
    HardTanh = 'aten::hardtanh'  # ReLU6(inplace=False)
    Int = 'aten::Int'
    List = 'prim::ListConstruct'
    Matmul = 'aten::matmul'
    MaxPool2D = 'aten::max_pool2d'
    NumToTensor = 'prim::NumToTensor'
    Param = 'prim::Param'
    Relu_ = 'aten::relu_'
    Relu = 'aten::relu'
    Reshape = 'aten::reshape'
    Size = 'aten::size'  # pytorch can get Tensor size dynamically
    T = 'aten::t'
    Tuple = 'prim::TupleConstruct'


class ConvParamIdx(object):
    input_tensor_idx = 0
    weight_idx = 1
    bias_idx = 2
    stride_idx = 3
    pad_idx = 4
    dilation_idx = 5
    transposed_idx = 6
    output_pad_idx = 7
    groups_idx = 8
    benchmark_idx = 9
    determinstic_idx = 10
    cudnn_enabled_idx = 11


class AvgPool2DParamIdx(object):
    input_tensor_idx = 0
    kernel_size_idx = 1
    stride_idx = 2
    pad_idx = 3
    ceil_mode_idx = 4
    count_include_pad_idx = 5
    divisor_override_idx = 6


class MaxPool2DParamIdx(object):
    input_tensor_idx = 0
    kernel_size_idx = 1
    stride_idx = 2
    pad_idx = 3
    dilation_idx = 4
    ceil_mode_idx = 5


class BNParamIdx(object):
    input_tensor_idx = 0
    weight_idx = 1
    bias_idx = 2
    running_mean_idx = 3
    running_var_idx = 4
    training_idx = 5
    momentum_idx = 6
    eps_idx = 7
    cudnn_enabled_idx = 8


class AddmmParamIdx(object):
    bias_idx = 0
    input_idx = 1
    weight_idx = 2
    beta_idx = 3
    alpha_idx = 4


class PytorchConverter(base_converter.ConverterInterface):
    '''A class for convert pytorch model to MACE model.'''
    activation_type = {
        'ReLU': ActivationType.RELU,
        'ReLU6': ActivationType.RELUX,
    }
    pooling_type_mode = {
        NodeKind.AvgPool2D: PoolingType.AVG,
        NodeKind.AdaptiveAvgPool2D: PoolingType.AVG,
        NodeKind.MaxPool2D: PoolingType.MAX,
    }
    eltwise_type = {
        NodeKind.Add: EltwiseType.SUM,
        NodeKind.Add_: EltwiseType.SUM,
    }

    def model_to_graph(self):
        dummy_input = ()
        for in_node in self._option.input_nodes.values():
            if len(in_node.shape) == 4:
                in_data_format = in_node.data_format
                if in_data_format == DataFormat.NHWC:
                    N, H, W, C = in_node.shape
                elif in_data_format == DataFormat.NCHW:
                    N, C, H, W = in_node.shape
                dummy_input = dummy_input + (torch.randn([N, C, H, W]),)
            else:
                dummy_input = dummy_input + (torch.randn(in_node.shape),)

        graph, params_dict = _model_to_graph(self._loaded_model, dummy_input)
        return graph, params_dict

    def init_output_shape_cache(self):
        self._output_shape_cache = {}
        for input_node in self._option.input_nodes.values():
            # input_shape is assigned in .yml file
            input_shape = input_node.shape[:]
            # transpose input from NHWC to NCHW
            if len(input_shape) == 4 and \
                    input_node.data_format == DataFormat.NHWC:
                Transformer.transpose_shape(input_shape, [0, 3, 1, 2])
            self._output_shape_cache[input_node.name] = input_shape

    def __init__(self, option, src_model_file):
        torch._C.Node.__getitem__ = _node_getitem
        self._param_converts = (
            NodeKind.Constant,
            NodeKind.List,
            NodeKind.Size,
            NodeKind.NumToTensor,
            NodeKind.Int,
        )

        self._option = option
        self._mace_net_def = mace_pb2.NetDef()
        ConverterUtil.set_filter_format(self._mace_net_def, DataFormat.OIHW)
        ConverterUtil.add_data_format_arg(self._mace_net_def, DataFormat.NCHW)
        ConverterUtil.set_framework_type(
            self._mace_net_def, FrameworkType.PYTORCH.value)
        self._op_converters = {
            NodeKind.AdaptiveAvgPool2D: self.convert_pool,
            NodeKind.Add: self.convert_add,
            NodeKind.Add_: self.convert_add,
            NodeKind.Addmm: self.convert_addmm,
            NodeKind.AvgPool2D: self.convert_pool,
            NodeKind.BatchNorm: self.convert_batch_norm,
            NodeKind.Cat: self.convert_cat,
            NodeKind.Convolution: self.convert_conv2d,
            NodeKind.Dropout: self.convert_dropout,
            NodeKind.Flatten: self.convert_flatten,
            NodeKind.HardTanh_: self.convert_hardtanh,
            NodeKind.HardTanh: self.convert_hardtanh,
            NodeKind.Matmul: self.convert_matmul,
            NodeKind.MaxPool2D: self.convert_pool,
            NodeKind.Relu: self.convert_relu,
            NodeKind.Relu_: self.convert_relu,
            NodeKind.Reshape: self.convert_reshape,
            NodeKind.T: self.convert_t,
        }
        self._loaded_model = torch.jit.load(src_model_file)
        self._loaded_model.eval()
        self._graph, self._params_dict = self.model_to_graph()
        self._output_node_name = list(self._graph.outputs())[0].debugName()
        self._output_value_type = list(self._graph.outputs())[0].type()
        mace_check(isinstance(self._output_value_type, (ValueType.TensorType,
                   ValueType.ListType, ValueType.TupleType)),
                   'return type {} not supported'.format(
                   self._output_value_type))
        self._node_map = {}
        self.init_output_shape_cache()

    def run(self):
        self.convert_ops()
        return self._mace_net_def

    def init_ignore_t(self, all_nodes):
        # ignore_t saves aten::t() which is ignored
        self.ignore_t = set()
        for node in all_nodes:
            node_kind = node.kind()
            if node_kind == NodeKind.T and self.is_trans_fc_w(node):
                self.ignore_t.add(node.output().debugName())

    def convert_ops(self):
        all_nodes = list(self._graph.nodes())
        self.init_ignore_t(all_nodes)
        for node in all_nodes:
            if isinstance(
                self._output_value_type,
                (ValueType.TupleType, ValueType.ListType)) and \
                    node.output().debugName() == self._output_node_name:
                print('pytorch return type is {},'
                      ' skipping adding it into MACE graph'.format(
                       self._output_value_type))
                continue

            inputs_vals = list(node.inputs())  # list of Value
            outputs_vals = list(node.outputs())
            mace_check(len(outputs_vals) == 1,
                       'pytorch converter supports nodes with single output,'
                       ' {} outputs found'.format(len(outputs_vals)))
            node_kind = node.kind()
            # This means this node is parameter for Ops
            if node_kind in self._param_converts:
                continue
            self._node_map[outputs_vals[0].debugName()] = node
            mace_check(node_kind in self._op_converters,
                       'MACE does not support pytorch node {} yet'.format(
                        node_kind))
            self._op_converters[node_kind](node, inputs_vals, outputs_vals)

    def convert_general_op(self, outputs_vals):
        op = self._mace_net_def.op.add()
        op.name = outputs_vals[0].debugName()

        data_type_arg = op.arg.add()
        data_type_arg.name = 'T'
        data_type_arg.i = self._option.data_type

        framework_type_arg = op.arg.add()
        framework_type_arg.name = MaceKeyword.mace_framework_type_str
        framework_type_arg.i = FrameworkType.PYTORCH.value

        ConverterUtil.add_data_format_arg(op, DataFormat.NCHW)

        return op

    def add_output_shape(self, op, shapes):
        mace_check(len(op.output) == len(shapes),
                   'Op {} ({}) output count is different from output shape count'.format(  # noqa
                    op.name, op.type))
        for i in range(len(shapes)):
            output_name = op.output[i]
            output_shape = op.output_shape.add()
            output_shape.dims.extend(shapes[i])
            self._output_shape_cache[output_name] = shapes[i]

    def infer_shape_general(self, op):
        if len(op.input) > 0:
            mace_check(op.input[0] in self._output_shape_cache,
                       "Op {} input {} does not exist".format(
                       op.name, op.input[0]))
            # initial values of _output_shape_cache come from:
            # 1: input_shape in .yml file(may be transposed to NCHW)
            # 2: tensor.dims. tensor is added by add_tensor_and_shape
            input_shape = self._output_shape_cache[op.input[0]]
            # pytorch BatchNorm 1/2/3D version use same function, the only way
            # to check  dimension of BatchNorm is to checkout input dimension
            if op.type == MaceOp.BatchNorm.name:
                mace_check(len(input_shape) == 4,
                           'only 2D BatchNorm is supported,'
                           ' but {}D input found'.format(len(input_shape)))
            self.add_output_shape(op, [input_shape])

    def infer_shape_conv2d_pool(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        output_shape = np.zeros_like(input_shape)
        if not op.type == MaceOp.Pooling:
            filter_shape = self._output_shape_cache[op.input[1]]
        paddings = ConverterUtil.get_arg(op,
                                         MaceKeyword.mace_padding_values_str).ints  # noqa
        strides = ConverterUtil.get_arg(op, MaceKeyword.mace_strides_str).ints
        dilations_arg = ConverterUtil.get_arg(op,
                                              MaceKeyword.mace_dilations_str)
        if dilations_arg is not None:
            dilations = dilations_arg.ints
        else:
            dilations = [1, 1]  # MACE pooling has no dilation
        if op.type == MaceOp.Pooling.name:
            kernels = ConverterUtil.get_arg(
                op, MaceKeyword.mace_kernel_str).ints
            if ConverterUtil.get_arg(
                    op, MaceKeyword.mace_global_pooling_str) is not None:
                kernels[0] = input_shape[2]
                kernels[1] = input_shape[3]

        round_func = math.floor
        round_mode_arg = ConverterUtil.get_arg(
            op, MaceKeyword.mace_round_mode_str)
        if round_mode_arg is not None and \
                round_mode_arg.i == RoundMode.CEIL.value:
            round_func = math.ceil

        # N_o = N_i
        output_shape[0] = input_shape[0]
        mace_check(ConverterUtil.data_format(op) == DataFormat.NCHW and
                   ConverterUtil.filter_format(self._mace_net_def) ==
                   DataFormat.OIHW, "MACE can only infer shape for NCHW input and OIHW filter")  # noqa
        # get C_{out}
        if op.type == MaceOp.DepthwiseConv2d.name:
            # 1CHW for depthwise, here C=C_{out}
            output_shape[1] = filter_shape[1]
        elif op.type == MaceOp.Conv2D.name:
            # filter format: OIHW
            output_shape[1] = filter_shape[0]
        else:
            output_shape[1] = input_shape[1]
        # H_{out}
        p, d, s = paddings[0], dilations[0], strides[0]
        k = kernels[0] if op.type == MaceOp.Pooling.name \
            else filter_shape[2]
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        output_shape[2] = int(
            round_func(float(input_shape[2] + p - d * (k - 1) - 1) /
                       float(s) + 1))
        # W_{out}
        p, d, s = paddings[1], dilations[1], strides[1]
        k = kernels[1] if op.type == MaceOp.Pooling.name \
            else filter_shape[3]
        output_shape[3] = int(
            round_func(float(input_shape[3] + p - d * (k - 1) - 1) /
                       float(s) + 1))

        self.add_output_shape(op, [output_shape])

    def convert_conv2d(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)

        op.input.extend([inputs_vals[i].debugName() for i in range(2)])
        op.output.extend([outputs_vals[0].debugName()])

        # OIHW
        key = inputs_vals[1].debugName()
        filter_shape = self._params_dict[key].shape
        filter_shape = [int(elem) for elem in filter_shape]  # Size -> list
        mace_check(len(filter_shape) == 4,
                   'MACE only supports 2D Conv, current Conv is {}D'.format(
                   len(filter_shape)-2))
        filter_cin = filter_shape[1]
        filter_cout = filter_shape[0]
        group_node = inputs_vals[ConvParamIdx.groups_idx].node()
        ngroups = group_node['value']
        mace_check(ngroups == 1 or ngroups == filter_cout,
                   'MACE only support conv without group or depthwise conv,'
                   ' but group number of {} found'.format(ngroups))
        is_depthwise = (ngroups != 1 and ngroups == filter_cout and
                        filter_cin == 1)
        if is_depthwise:
            op.type = MaceOp.DepthwiseConv2d.name
        else:
            op.type = MaceOp.Conv2D.name

        strides_node = inputs_vals[ConvParamIdx.stride_idx].node()
        strides_vals = list(strides_node.inputs())
        mace_strides = [strides_vals[i].node()['value'] for i in range(2)]
        strides_arg = op.arg.add()
        strides_arg.name = MaceKeyword.mace_strides_str
        strides_arg.ints.extend(mace_strides)

        pads_node = inputs_vals[ConvParamIdx.pad_idx].node()
        pads_vals = list(pads_node.inputs())
        mace_pads = [2 * pads_vals[i].node()['value'] for i in range(2)]
        pads_arg = op.arg.add()
        pads_arg.name = MaceKeyword.mace_padding_values_str
        pads_arg.ints.extend(mace_pads)

        dilations_node = inputs_vals[ConvParamIdx.dilation_idx].node()
        dilations_vals = list(dilations_node.inputs())
        mace_dilations = [dilations_vals[i].node()['value'] for i in range(2)]
        dilation_arg = op.arg.add()
        dilation_arg.name = MaceKeyword.mace_dilations_str
        dilation_arg.ints.extend(mace_dilations)

        filter_tensor_name = inputs_vals[ConvParamIdx.weight_idx].debugName()
        filter_data = self._params_dict[filter_tensor_name]
        if is_depthwise:
            # C1HW => 1CHW
            filter_data = filter_data.permute((1, 0, 2, 3))
        filter_data = filter_data.numpy()
        self.add_tensor_and_shape(filter_tensor_name, filter_data.shape,
                                  mace_pb2.DT_FLOAT, filter_data)

        bias_val = inputs_vals[ConvParamIdx.bias_idx]
        has_bias = (not isinstance(bias_val.type(), ValueType.NoneType))
        if has_bias:
            bias_tensor_name = inputs_vals[ConvParamIdx.bias_idx].debugName()
            bias_data = self._params_dict[bias_tensor_name]
            bias_data = bias_data.numpy()
            self.add_tensor_and_shape(bias_tensor_name, bias_data.shape,
                                      mace_pb2.DT_FLOAT, bias_data)
            op.input.extend([bias_tensor_name])
        self.infer_shape_conv2d_pool(op)

    def convert_batch_norm(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)

        op.input.extend([inputs_vals[0].debugName()])
        op.output.extend([outputs_vals[0].debugName()])
        op.type = MaceOp.BatchNorm.name

        is_training = int(inputs_vals[BNParamIdx.training_idx].node()['value'])
        mace_check(is_training == 0,
                   "Only support batch normalization with is_training = 0,"
                   " but got {}".format(is_training))
        state_dict = self._params_dict
        gamma_key = inputs_vals[BNParamIdx.weight_idx].debugName()
        gamma_value = state_dict[gamma_key].numpy().astype(np.float32)
        beta_key = inputs_vals[BNParamIdx.bias_idx].debugName()
        beta_value = state_dict[beta_key].numpy().astype(np.float32)
        mean_name = inputs_vals[BNParamIdx.running_mean_idx].debugName()
        mean_value = state_dict[mean_name].numpy().astype(np.float32)
        var_name = inputs_vals[BNParamIdx.running_var_idx].debugName()
        var_value = state_dict[var_name].numpy().astype(np.float32)
        epsilon_value = inputs_vals[BNParamIdx.eps_idx].node()['value']

        scale_name = gamma_key + '_scale'
        offset_name = beta_key + '_offset'

        scale_value = (
            (1.0 / np.vectorize(math.sqrt)(
             var_value + epsilon_value)) * gamma_value)
        offset_value = (-mean_value * scale_value) + beta_value

        self.add_tensor_and_shape(scale_name, scale_value.shape,
                                  mace_pb2.DT_FLOAT, scale_value)
        self.add_tensor_and_shape(offset_name, offset_value.shape,
                                  mace_pb2.DT_FLOAT, offset_value)
        op.input.extend([scale_name, offset_name])
        self.infer_shape_general(op)

    def convert_hardtanh(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)
        op.type = MaceOp.Activation.name

        min_val = inputs_vals[1].node()['value']
        max_val = inputs_vals[2].node()['value']
        mace_check(abs(min_val) < 1e-8, "MACE only supports min == 0 Clip op")

        op.input.extend([inputs_vals[0].debugName()])
        op.output.extend([outputs_vals[0].debugName()])

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_activation_type_str
        mace_check(abs(max_val - 6.) < 1e-8,
                   'only support converting hardtanh_ to ReLU6 yet')
        type_arg.s = six.b(self.activation_type['ReLU6'].name)

        limit_arg = op.arg.add()
        limit_arg.name = MaceKeyword.mace_activation_max_limit_str
        limit_arg.f = 6.0
        self.infer_shape_general(op)

    def convert_add(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)
        op.type = MaceOp.Eltwise.name
        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_element_type_str
        node_kind = node.kind()
        type_arg.i = self.eltwise_type[node_kind].value
        alpha = inputs_vals[2].node()['value']
        mace_check(alpha == 1,
                   'MACE only support alpha value of 1 for Add Op,'
                   ' {} found'.format(alpha))
        op.input.extend([inputs_vals[i].debugName() for i in range(2)])
        op.output.extend([outputs_vals[0].debugName()])
        lhs_kind = inputs_vals[0].node().kind()
        rhs_kind = inputs_vals[1].node().kind()
        if lhs_kind != NodeKind.Constant and rhs_kind == NodeKind.Constant:
            const_value = inputs_vals[1].node()['value']
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = float(const_value)
            value_index_arg = op.arg.add()
            value_index_arg.name = MaceKeyword.mace_scalar_input_index_str
            value_index_arg.i = 1
            del op.input[1]
        elif lhs_kind == NodeKind.Constant and rhs_kind != NodeKind.Constant:
            const_value = inputs_vals[0].node()['value']
            value_arg = op.arg.add()
            value_arg.name = MaceKeyword.mace_scalar_input_str
            value_arg.f = float(const_value)
            value_index_arg = op.arg.add()
            value_index_arg.name = MaceKeyword.mace_scalar_input_index_str
            value_index_arg.i = 0
            del op.input[0]
        self.infer_shape_general(op)

    def convert_relu(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)
        op.type = MaceOp.Activation.name

        op.input.extend([inputs_vals[0].debugName()])
        op.output.extend([outputs_vals[0].debugName()])

        type_arg = op.arg.add()
        type_arg.name = MaceKeyword.mace_activation_type_str
        type_arg.s = six.b(self.activation_type['ReLU'].name)

        self.infer_shape_general(op)

    def infer_shape_cat(self, op):
        output_shape = list(self._output_shape_cache[op.input[0]])
        axis = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str).i
        if axis < 0:
            axis = len(output_shape) + axis
        output_shape[axis] = 0
        for input_node in op.input:
            input_shape = list(self._output_shape_cache[input_node])
            output_shape[axis] = output_shape[axis] + input_shape[axis]
        self.add_output_shape(op, [output_shape])

    def convert_cat(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)
        op.type = MaceOp.Concat.name
        in_vals = list(inputs_vals[0].node().inputs())
        in_names = [in_vals[i].debugName() for i in range(len(in_vals))]
        op.input.extend(in_names)
        op.output.extend([outputs_vals[0].debugName()])

        axis_int = inputs_vals[1].node()['value']
        axis_arg = op.arg.add()
        axis_arg.name = MaceKeyword.mace_axis_str
        axis_arg.i = axis_int

        self.infer_shape_cat(op)

    def convert_flatten(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)
        op.type = MaceOp.Reshape.name
        op.input.extend([inputs_vals[0].debugName()])
        op.output.extend([outputs_vals[0].debugName()])

        input_shape = list(self._output_shape_cache[op.input[0]])
        ndim = len(input_shape)
        start_dim = inputs_vals[1].node()['value']
        if start_dim < 0:
            start_dim += ndim
        end_dim = inputs_vals[2].node()['value']
        if end_dim < 0:
            end_dim += ndim
        reshape_dims = []
        for i in range(0, start_dim):
            reshape_dims.append(input_shape[i])
        mid_shape = 1
        for i in range(start_dim, end_dim+1):
            mid_shape *= input_shape[i]
        reshape_dims.append(mid_shape)
        for i in range(end_dim + 1, ndim):
            reshape_dims.append(input_shape[i])

        dim_arg = op.arg.add()
        dim_arg.name = MaceKeyword.mace_dim_str
        dim_arg.ints.extend(reshape_dims)
        self.infer_shape_reshape(op)

    def get_weight_from_node(self, node):
        input_list = list(node.inputs())
        key = input_list[0].debugName()
        return self._params_dict[key]

    def is_trans_fc_w(self, node):
        in_vals = list(node.inputs())
        mace_check(len(in_vals) == 1, 't() must have 1 input')
        in_name = in_vals[0].debugName()
        if in_name in self._params_dict and \
                len(self._params_dict[in_name].shape) == 2:
            return True
        return False

    def infer_shape_fully_connected(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        weight_shape = self._output_shape_cache[op.input[1]]
        data_format = ConverterUtil.data_format(op)
        mace_check(data_format == DataFormat.NCHW,
                   "format {} is not supported".format(data_format))
        output_shape = [input_shape[0], weight_shape[0], 1, 1]
        self.add_output_shape(op, [output_shape])

    def convert_addmm(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)

        weight_in_node = inputs_vals[AddmmParamIdx.weight_idx].node()
        is_mat2_w = weight_in_node.kind() == NodeKind.T and self.is_trans_fc_w(
            weight_in_node)
        alpha = inputs_vals[AddmmParamIdx.alpha_idx].node()['value']
        alpha_type = inputs_vals[AddmmParamIdx.alpha_idx].type()
        is_alpha_fc = isinstance(alpha_type, ValueType.IntType) and alpha == 1
        is_bias_w = inputs_vals[AddmmParamIdx.bias_idx].debugName() in \
            self._params_dict
        beta = inputs_vals[AddmmParamIdx.beta_idx].node()['value']
        beta_type = inputs_vals[AddmmParamIdx.beta_idx].type()
        is_beta_fc = isinstance(beta_type, ValueType.IntType) and beta == 1

        # when mat2 is from state_dict and alpha=1 and bias is from state_dict
        # and beta =1, it is fc
        is_fc = is_mat2_w and is_alpha_fc and is_bias_w and is_beta_fc

        mace_check(is_fc, 'addmm can only be converted into FC yet')
        # pytorch usually prepend a reshape/flatten before fc, convert fc
        # into matmul followed by biasadd, thus reshape/flatten and matmul
        # will be merged. see transform_matmul_to_fc for detail.
        name_back = op.name
        matmul_op_name = op.name + '_matmul'
        op.name = matmul_op_name
        op.type = MaceOp.MatMul.name
        fc_upstream_name = inputs_vals[AddmmParamIdx.input_idx].debugName()
        op.input.extend([fc_upstream_name])
        op.output.extend([matmul_op_name])

        weight_tensor_name = op.name + '_weight'
        weight_tensor = self.get_weight_from_node(weight_in_node)
        weight_data = weight_tensor.numpy()
        self.add_tensor_and_shape(weight_tensor_name, weight_data.shape,
                                  mace_pb2.DT_FLOAT, weight_data)
        op.input.extend([weight_tensor_name])

        transpose_a_arg = op.arg.add()
        transpose_a_arg.name = MaceKeyword.mace_transpose_a_str
        transpose_a_arg.i = 0
        transpose_b_arg = op.arg.add()
        transpose_b_arg.name = MaceKeyword.mace_transpose_b_str
        transpose_b_arg.i = 1  # OxI, trans_b needed

        self.infer_shape_matmul(op)

        opb = self.convert_general_op(outputs_vals)
        opb.type = MaceOp.BiasAdd.name
        bias_tensor_name = opb.name + '_bias'
        key = inputs_vals[AddmmParamIdx.bias_idx].debugName()
        bias_data = self._params_dict[key]
        bias_data = bias_data.numpy()
        self.add_tensor_and_shape(bias_tensor_name, bias_data.reshape(
            -1).shape, mace_pb2.DT_FLOAT, bias_data)
        opb.input.extend([matmul_op_name, bias_tensor_name])
        opb.output.extend([name_back])
        self.infer_shape_general(opb)

    def infer_shape_matmul(self, op):
        lhs_shape = self._output_shape_cache[op.input[0]]
        lhs_rank = len(lhs_shape)
        lhs_rows = lhs_shape[-2]
        lhs_cols = lhs_shape[-1]
        rhs_shape = self._output_shape_cache[op.input[1]]
        rhs_rank = len(rhs_shape)
        rhs_rows = rhs_shape[-2]
        rhs_cols = rhs_shape[-1]
        transpose_a_ = ConverterUtil.get_arg(
            op, MaceKeyword.mace_transpose_a_str).i
        transpose_b_ = ConverterUtil.get_arg(
            op, MaceKeyword.mace_transpose_b_str).i

        rows = lhs_cols if transpose_a_ else lhs_rows
        cols = rhs_rows if transpose_b_ else rhs_cols

        if lhs_rank >= rhs_rank:
            if lhs_rank > rhs_rank:
                mace_check(rhs_rank == 2,
                           'The rhs rank of non-batched MatMul must be 2')  # noqa
            output_shape = lhs_shape.copy()
            output_shape[lhs_rank - 2] = rows
            output_shape[lhs_rank - 1] = cols
        else:
            output_shape = rhs_shape.copy()
            output_shape[rhs_rank - 2] = rows
            output_shape[rhs_rank - 1] = cols
        self.add_output_shape(op, [output_shape])

    def convert_matmul(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)

        weight_in_node = inputs_vals[1].node()
        is_weight = weight_in_node.kind() == NodeKind.T and self.is_trans_fc_w(
            weight_in_node)

        op.type = MaceOp.MatMul.name
        op.input.extend([inputs_vals[i].debugName() for i in range(2)])
        op.output.extend([outputs_vals[0].debugName()])
        if is_weight:
            weight_tensor_name = op.input[1]
            weight_val = inputs_vals[1]
            weight_tensor = self.get_weight_from_node(weight_in_node)
            weight_data = weight_tensor.numpy()
            self.add_tensor_and_shape(weight_tensor_name, weight_data.shape,
                                      mace_pb2.DT_FLOAT, weight_data)

        lhs_shape = self._output_shape_cache[op.input[0]]
        rhs_shape = self._output_shape_cache[op.input[1]]
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)
        mace_check(lhs_rank >= 2 and rhs_rank >= 2,
                   "The rank of MatMul must be >= 2,"
                   " but lhs_rank = {} and rhs_rank = {} found".format(
                    lhs_rank, rhs_rank))

        transpose_a_arg = op.arg.add()
        transpose_a_arg.name = MaceKeyword.mace_transpose_a_str
        transpose_a_arg.i = 0
        transpose_b_arg = op.arg.add()
        transpose_b_arg.name = MaceKeyword.mace_transpose_b_str
        if is_weight:
            transpose_b_arg.i = 1
        else:
            transpose_b_arg.i = 0

        self.infer_shape_matmul(op)

    def convert_pool(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)
        op.type = MaceOp.Pooling.name

        op.input.extend([inputs_vals[0].debugName()])
        op.output.extend([outputs_vals[0].debugName()])

        node_kind = node.kind()
        idx_map = {NodeKind.AvgPool2D: AvgPool2DParamIdx,
                   NodeKind.MaxPool2D: MaxPool2DParamIdx}

        if node_kind == NodeKind.AdaptiveAvgPool2D:
            output_shape_node = inputs_vals[1].node()
            output_shape_vals = list(output_shape_node.inputs())
            target_output_shape = [
                output_shape_vals[i].node()['value'] for i in range(2)]
            mace_check(target_output_shape[0] == 1 and
                       target_output_shape[1] == 1,
                       'only support output shape of [1, 1] for AdaptiveAvgPool2D')  # noqa

            strides_arg = op.arg.add()
            strides_arg.name = MaceKeyword.mace_strides_str
            strides_arg.ints.extend([1, 1])

            pads_arg = op.arg.add()
            pads_arg.name = MaceKeyword.mace_padding_values_str
            pads_arg.ints.extend([0, 0])

            kernels_arg = op.arg.add()
            kernels_arg.name = MaceKeyword.mace_kernel_str
            kernels_arg.ints.extend([0, 0])

            global_pooling_arg = op.arg.add()
            global_pooling_arg.name = MaceKeyword.mace_global_pooling_str
            global_pooling_arg.i = 1
        else:
            pad_node = inputs_vals[idx_map[node_kind].pad_idx].node()
            pad_vals = list(pad_node.inputs())
            mace_check(len(pad_vals) == 2,
                       "only support 2D pooling,"
                       " but {}D padding value found".format(len(pad_vals)))
            # MACE pads include both sides, pytorch pads include single side
            pads = [2 * pad_vals[i].node()['value'] for i in range(2)]
            pads_arg = op.arg.add()
            pads_arg.name = MaceKeyword.mace_padding_values_str
            pads_arg.ints.extend(pads)

            if node_kind == NodeKind.MaxPool2D:
                dilation_node = inputs_vals[
                    idx_map[node_kind].dilation_idx].node()
                dilation_vals = list(dilation_node.inputs())
                dilations = [dilation_vals[i].node()['value']
                             for i in range(2)]
                mace_check(dilations[0] == 1 and dilations[1] == 1,
                           "MACE pooling does not support dilation")

            kernel_node = inputs_vals[
                idx_map[node_kind].kernel_size_idx].node()
            kernel_vals = list(kernel_node.inputs())
            kernels = [kernel_vals[i].node()['value'] for i in range(2)]
            kernels_arg = op.arg.add()
            kernels_arg.name = MaceKeyword.mace_kernel_str
            kernels_arg.ints.extend(kernels)

            stride_node = inputs_vals[idx_map[node_kind].stride_idx].node()
            stride_vals = list(stride_node.inputs())
            strides = [stride_vals[i].node()['value'] for i in range(2)]
            strides_arg = op.arg.add()
            strides_arg.name = MaceKeyword.mace_strides_str
            strides_arg.ints.extend(strides)

            ceil_node = inputs_vals[idx_map[node_kind].ceil_mode_idx].node()
            ceil_mode = bool(ceil_node['value'])
            round_mode_arg = op.arg.add()
            round_mode_arg.name = MaceKeyword.mace_round_mode_str
            round_mode_arg.i = RoundMode.FLOOR.value
            if ceil_mode:
                round_mode_arg.i = RoundMode.CEIL.value

            if node_kind == NodeKind.AvgPool2D:
                count_include_pad_node = inputs_vals[
                    AvgPool2DParamIdx.count_include_pad_idx].node()
                count_include_pad = bool(count_include_pad_node['value'])
                # if there is no padding, count_include_pad has no effect;
                # otherwise, MACE does not support count_include_pad.
                if count_include_pad:
                    mace_check(pads[0] == 0 and pads[1] == 0,
                               "if count_include_pad is set, pad must be zero."
                               " pad values ({},{}) found.".format(
                               pads[0], pads[1]))
                # divisor
                divisor_override_node = inputs_vals[
                    AvgPool2DParamIdx.divisor_override_idx].node()
                mace_check(isinstance(divisor_override_node.output().type(),
                           ValueType.NoneType),
                           "MACE does not support divisor_override parameter for AvgPool2D")  # noqa

        pooling_type_arg = op.arg.add()
        pooling_type_arg.name = MaceKeyword.mace_pooling_type_str
        pooling_type_arg.i = self.pooling_type_mode[node_kind].value

        self.infer_shape_conv2d_pool(op)

    def node_to_int(self, shape_node):
        if shape_node.kind() == NodeKind.Constant:
            return shape_node['value']
        elif shape_node.kind() == NodeKind.Size:
            input_node = list(shape_node.inputs())[0].node()
            axis_node = list(shape_node.inputs())[1].node()
            axis_int = self.node_to_int(axis_node)
            input_tensor_shape = self._output_shape_cache[
                input_node.output().debugName()]
            return input_tensor_shape[axis_int]
        else:
            input_node = list(shape_node.inputs())[0].node()
            return self.node_to_int(input_node)

    def infer_shape_reshape(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        dim_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_dim_str)
        output_shape = list(dim_arg.ints)  # reshape_dims
        product = input_size = 1
        idx = -1
        num_minus1 = 0

        for dim in input_shape:
            input_size *= dim
        reshape_dims = list(dim_arg.ints)
        for i in range(len(reshape_dims)):
            if reshape_dims[i] == -1:
                idx = i
                output_shape[i] = 1
                num_minus1 += 1
            else:
                output_shape[i] = reshape_dims[i]
                product *= reshape_dims[i]
        mace_check(num_minus1 <= 1, 'only 0 or 1 negative shape supported')
        if idx != -1:
            output_shape[idx] = int(input_size / product)

        self.add_output_shape(op, [output_shape])

    def convert_reshape(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)
        op.type = MaceOp.Reshape.name
        op.input.extend([inputs_vals[0].debugName()])
        op.output.extend([outputs_vals[0].debugName()])

        # sometimes pytorch's reshape parameter is infered from input Tensor,
        # e.g. y = y.reshape(x.size(0), -1)
        # so static parameter may be unable to get here.
        dim_arg = op.arg.add()
        dim_arg.name = MaceKeyword.mace_dim_str
        shape_list_node = list(node.inputs())[1].node()
        reshape_dims = []
        for shape_val in shape_list_node.inputs():
            shape_node = shape_val.node()
            _kind = shape_node.kind()
            if _kind == NodeKind.Constant:
                reshape_dims.append(shape_node['value'])
            elif _kind == NodeKind.Int:
                _dim = int(self.node_to_int(shape_node))
                reshape_dims.append(_dim)
            else:
                print('unsupported shape node kind {}'.format(_kind))
        dim_arg.ints.extend(reshape_dims)
        self.infer_shape_reshape(op)

    def infer_shape_identity(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        output_shape = input_shape
        self.add_output_shape(op, [output_shape])

    def convert_dropout(self, node, inputs_vals, outputs_vals):
        op = self.convert_general_op(outputs_vals)
        training = int(inputs_vals[2].node()['value'])
        mace_check(training == 0, 'for inference, dropout must be disabled')
        op.type = MaceOp.Identity.name
        op.input.extend([inputs_vals[0].debugName()])
        op.output.extend([outputs_vals[0].debugName()])
        self.infer_shape_identity(op)

    def infer_shape_transpose(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        output_shape = np.zeros(len(input_shape), dtype=np.int32)
        dims_arg = ConverterUtil.get_arg(op, MaceKeyword.mace_dims_str)
        dims_ints = dims_arg.ints
        for idx in range(len(dims_ints)):
            output_shape[idx] = input_shape[dims_ints[idx]]
        self.add_output_shape(op, [output_shape])

    def convert_t(self, node, inputs_vals, outputs_vals):
        if node.output().debugName() in self.ignore_t:
            return
        op = self.convert_general_op(outputs_vals)

        op.input.extend([inputs_vals[0].debugName()])
        op.output.extend([outputs_vals[0].debugName()])

        # MACE only supports transpose 2/3/4D tensor.
        input_shape = self._output_shape_cache[op.input[0]]
        if len(input_shape) <= 1:
            op.type = MaceOp.Identity.name
            self.infer_shape_general(op)
        else:
            op.type = MaceOp.Transpose.name
            dims_arg = op.arg.add()
            dims_arg.name = MaceKeyword.mace_dims_str
            dims_arg.ints.extend([1, 0])
            self.infer_shape_transpose(op)

    def add_tensor_and_shape(self, name, shape, data_type, value):
        tensor = self._mace_net_def.tensors.add()
        tensor.name = name
        tensor.dims.extend(list(shape))
        tensor.data_type = data_type
        tensor.float_data.extend(value.flat)
        self._output_shape_cache[name] = np.array(shape)
