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


import math
import numpy as np

from mace.python.tools.converter_tool.transformer import Transformer
from mace.python.tools.converter_tool.base_converter import DataFormat
from mace.python.tools.converter_tool.base_converter import FilterFormat
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.convert_util import mace_check


class ShapeInference(object):
    """Currently we only use it to infer caffe shape, we use tensorflow engine
    to infer tensorflow op shapes, since tensorflow has too many ops."""

    def __init__(self, net, input_nodes):
        self._op_shape_inference = {
            MaceOp.Conv2D.name: self.infer_shape_conv_pool_shape,
            MaceOp.Deconv2D.name: self.infer_shape_deconv,
            MaceOp.DepthwiseConv2d.name: self.infer_shape_conv_pool_shape,
            MaceOp.Eltwise.name: self.infer_shape_general,
            MaceOp.FoldedBatchNorm.name: self.infer_shape_general,
            MaceOp.AddN.name: self.infer_shape_general,
            MaceOp.Activation.name: self.infer_shape_general,
            MaceOp.Pooling.name: self.infer_shape_conv_pool_shape,
            MaceOp.Concat.name: self.infer_shape_concat,
            MaceOp.Split.name: self.infer_shape_slice,
            MaceOp.Softmax.name: self.infer_shape_general,
            MaceOp.FullyConnected.name: self.infer_shape_fully_connected,
            MaceOp.Crop.name: self.infer_shape_crop,
        }

        self._net = net
        self._output_shape_cache = {}
        for input_node in input_nodes:
            input_shape = input_node.shape[:]
            # transpose input from NCHW to NHWC
            Transformer.transpose_shape(input_shape, [0, 3, 1, 2])
            self._output_shape_cache[input_node.name] = input_shape
        for tensor in net.tensors:
            self._output_shape_cache[tensor.name] = list(tensor.dims)

    def run(self):
        for op in self._net.op:
            mace_check(op.type in self._op_shape_inference,
                       "Mace does not support caffe op type %s yet"
                       % op.type)
            self._op_shape_inference[op.type](op)

    def add_output_shape(self, op, shapes):
        mace_check(len(op.output) == len(shapes),
                   "Op %s (%s) output count is different from "
                   "output shape count" % (
                       op.name, op.type))
        for i in xrange(len(shapes)):
            output_name = op.output[i]
            output_shape = op.output_shape.add()
            output_shape.dims.extend(shapes[i])
            self._output_shape_cache[output_name] = shapes[i]

    def infer_shape_general(self, op):
        if len(op.input) > 0:
            mace_check(op.input[0] in self._output_shape_cache,
                       "%s does not exist" % op.input[0])
            input_shape = self._output_shape_cache[op.input[0]]
            self.add_output_shape(op, [input_shape])

    def infer_shape_conv_pool_shape(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        output_shape = np.zeros_like(input_shape)
        if op.type == MaceOp.Pooling:
            filter_shape = list(
                ConverterUtil.get_arg(op, MaceKeyword.mace_kernel_str).ints)
            if ConverterUtil.data_format(op) == DataFormat.NCHW:
                filter_shape = [input_shape[1], input_shape[1]] + filter_shape
                if ConverterUtil.get_arg(op,
                                         MaceKeyword.mace_global_pooling_str) \
                        is not None:
                    filter_shape[2] = input_shape[2]
                    filter_shape[3] = input_shape[3]
            else:  # NHWC
                filter_shape = filter_shape + [input_shape[1], input_shape[1]]
                if ConverterUtil.get_arg(op,
                                         MaceKeyword.mace_global_pooling_str) \
                        is not None:
                    filter_shape[0] = input_shape[1]
                    filter_shape[1] = input_shape[2]
        else:
            filter_shape = self._output_shape_cache[op.input[1]]

        paddings = ConverterUtil.get_arg(op,
                                         MaceKeyword.mace_padding_values_str).ints  # noqa
        strides = ConverterUtil.get_arg(op, MaceKeyword.mace_strides_str).ints
        dilations_arg = ConverterUtil.get_arg(op,
                                              MaceKeyword.mace_dilations_str)
        if dilations_arg is not None:
            dilations = dilations_arg.ints
        else:
            dilations = [1, 1]
        if op.type == MaceOp.Pooling:
            round_func = math.ceil
        else:
            round_func = math.floor

        output_shape[0] = input_shape[0]
        if ConverterUtil.data_format(op) == DataFormat.NCHW \
                and ConverterUtil.filter_format(self._net) == FilterFormat.OIHW:  # noqa
            # filter format: OIHW
            if op.type == MaceOp.DepthwiseConv2d.name:
                output_shape[1] = filter_shape[0] * filter_shape[1]
            else:
                output_shape[1] = filter_shape[0]
            output_shape[2] = int(
                round_func((input_shape[2] + paddings[0] - filter_shape[2] -
                            (filter_shape[2] - 1) *
                            (dilations[0] - 1)) / float(strides[0]))) + 1
            output_shape[3] = int(
                round_func((input_shape[3] + paddings[1] - filter_shape[3] -
                            (filter_shape[3] - 1) *
                            (dilations[1] - 1)) / float(strides[1]))) + 1
        else:
            mace_check(False,
                       "Mace can only infer shape for"
                       " NCHW input and OIHW filter")

        self.add_output_shape(op, [output_shape])

    def infer_shape_deconv(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        output_shape = np.zeros_like(input_shape)
        filter_shape = self._output_shape_cache[op.input[1]]

        paddings = ConverterUtil.get_arg(op,
                                         MaceKeyword.mace_padding_values_str).ints  # noqa
        strides = ConverterUtil.get_arg(op, MaceKeyword.mace_strides_str).ints
        dilations_arg = ConverterUtil.get_arg(op,
                                              MaceKeyword.mace_dilations_str)
        if dilations_arg is not None:
            dilations = dilations_arg.ints
        else:
            dilations = [1, 1]
        round_func = math.floor

        output_shape[0] = input_shape[0]
        if ConverterUtil.data_format(op) == DataFormat.NCHW \
                and ConverterUtil.filter_format(self._net) == FilterFormat.OIHW:  # noqa
            # filter format: IOHW
            output_shape[1] = filter_shape[1]
            output_shape[2] = int(
                round_func((input_shape[2] - 1) * strides[0] +
                           (filter_shape[2] - 1) * (dilations[0] - 1) +
                           filter_shape[2] - paddings[0]))
            output_shape[3] = int(
                round_func((input_shape[3] - 1) * strides[1] +
                           (filter_shape[3] - 1) * (dilations[1] - 1) +
                           filter_shape[3] - paddings[1]))
        else:
            mace_check(False,
                       "Mace can only infer shape for"
                       " NCHW input and OIHW filter")
        print ("deconv layer %s (%s) input:%s filter:%s output:%s" %
               (op.name, op.type, input_shape, filter_shape, output_shape))

        self.add_output_shape(op, [output_shape])

    def infer_shape_concat(self, op):
        output_shape = self._output_shape_cache[op.input[0]]
        axis = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str).i
        for input_node in op.input:
            input_shape = self._output_shape_cache[input_node]
            output_shape[axis] += input_shape[axis]

        self.add_output_shape(op, [output_shape])

    def infer_shape_slice(self, op):
        output_shape = self._output_shape_cache[op.input[0]]
        axis = ConverterUtil.get_arg(op, MaceKeyword.mace_axis_str).i
        output_shape[axis] /= len(op.output)
        output_shapes = []
        for _ in op.output:
            output_shapes.append(output_shape)
        self.add_output_shape(op, output_shapes)

    def infer_shape_fully_connected(self, op):
        input_shape = self._output_shape_cache[op.input[0]]
        weight_shape = self._output_shape_cache[op.input[1]]
        if ConverterUtil.data_format(op) == DataFormat.NCHW:
            output_shape = [input_shape[0], weight_shape[0], 1, 1]
        else:
            mace_check(False, "format %s is not supported"
                       % ConverterUtil.data_format(op))
        self.add_output_shape(op, [output_shape])

    def infer_shape_crop(self, op):
        mace_check(len(op.input) == 2, "crop layer needs two inputs")
        output_shape = self._output_shape_cache[op.input[1]]
        self.add_output_shape(op, [output_shape])
