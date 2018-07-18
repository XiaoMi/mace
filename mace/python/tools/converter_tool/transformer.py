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


import enum
import numpy as np

from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter
from mace.python.tools.converter_tool.base_converter import ActivationType
from mace.python.tools.converter_tool.base_converter import ConverterUtil
from mace.python.tools.converter_tool.base_converter import DataFormat
from mace.python.tools.converter_tool.base_converter import DeviceType
from mace.python.tools.converter_tool.base_converter import EltwiseType
from mace.python.tools.converter_tool.base_converter import FilterFormat
from mace.python.tools.converter_tool.base_converter import MaceKeyword
from mace.python.tools.converter_tool.base_converter import MaceOp
from mace.python.tools.converter_tool.base_converter import PaddingMode
from mace.python.tools.converter_tool.base_converter import TransformerRule
from mace.python.tools.convert_util import mace_check

OPENCL_IMAGE_MAX_SIZE = 16384


class OpenCLBufferType(enum.Enum):
    CONV2D_FILTER = 0
    IN_OUT_CHANNEL = 1
    ARGUMENT = 2
    IN_OUT_HEIGHT = 3
    IN_OUT_WIDTH = 4
    WINOGRAD_FILTER = 5
    DW_CONV2D_FILTER = 6
    WEIGHT_HEIGHT = 7
    WEIGHT_WIDTH = 8


class Transformer(base_converter.ConverterInterface):
    """A class for transform naive mace model to optimized model.
    This Transformer should be platform irrelevant. So, do not assume
    tensor name has suffix like ':0".
    """

    def __init__(self, option, model):
        # Dependencies
        # (TRANSFORM_MATMUL_TO_FC, TRANSFORM_GLOBAL_CONV_TO_FC) -> RESHAPE_FC_WEIGHT  # noqa
        self._registered_transformers = {
            TransformerRule.REMOVE_IDENTITY_OP: self.remove_identity_op,
            TransformerRule.TRANSFORM_GLOBAL_POOLING:
                self.transform_global_pooling,
            TransformerRule.FOLD_RESHAPE: self.fold_reshape,
            TransformerRule.TRANSFORM_MATMUL_TO_FC:
                self.transform_matmul_to_fc,
            TransformerRule.FOLD_BATCHNORM: self.fold_batchnorm,
            TransformerRule.FOLD_CONV_AND_BN:
                self.fold_conv_and_bn,  # data_format related
            TransformerRule.FOLD_DEPTHWISE_CONV_AND_BN:
                self.fold_depthwise_conv_and_bn,  # data_format related
            TransformerRule.TRANSFORM_GPU_WINOGRAD:
                self.transform_gpu_winograd,  # data_format related
            TransformerRule.TRANSFORM_ADD_TO_BIASADD:
                self.transform_add_to_biasadd,
            TransformerRule.FOLD_BIASADD: self.fold_biasadd,
            TransformerRule.FLATTEN_ATROUS_CONV: self.flatten_atrous_conv,
            TransformerRule.FOLD_ACTIVATION: self.fold_activation,
            TransformerRule.TRANSPOSE_FILTERS: self.transpose_filters,
            TransformerRule.TRANSPOSE_DATA_FORMAT: self.transpose_data_format,
            TransformerRule.ADD_IN_OUT_TENSOR_INFO:
                self.add_in_out_tensor_info,
            TransformerRule.TRANSFORM_GLOBAL_CONV_TO_FC:
                self.transform_global_conv_to_fc,
            TransformerRule.RESHAPE_FC_WEIGHT: self.reshape_fc_weight,
            TransformerRule.TRANSFORM_BUFFER_IMAGE:
                self.transform_buffer_image,
            TransformerRule.ADD_DEVICE:
                self.add_device,
            TransformerRule.UPDATE_FLOAT_OP_DATA_TYPE:
                self.update_float_op_data_type,
            TransformerRule.ADD_MACE_INPUT_AND_OUTPUT_NODES:
                self.add_mace_input_and_output_nodes,
            TransformerRule.SORT_BY_EXECUTION: self.sort_by_execution,
        }

        self._option = option
        self._model = model
        self._gpu_wino_blk = self._option.winograd

        self._ops = {}
        self._consts = {}
        self._consumers = {}
        self._producer = {}
        self._target_data_format = DataFormat.NHWC
        self._input_output_added = False

        if self._option.device == DeviceType.CPU.value:
            self._target_data_format = DataFormat.NCHW

    def run(self):
        for key in self._option.transformer_option:
            transformer = self._registered_transformers[key]
            while True:
                self.construct_ops_and_consumers()
                changed = transformer()
                if not changed:
                        break

        return self._model

    def filter_format(self):
        filter_format_value = ConverterUtil.get_arg(self._model,
                                                    MaceKeyword.mace_filter_format_str).i  # noqa
        filter_format = None
        if filter_format_value == FilterFormat.HWIO.value:
            filter_format = FilterFormat.HWIO
        elif filter_format_value == FilterFormat.OIHW.value:
            filter_format = FilterFormat.OIHW
        elif filter_format_value == FilterFormat.HWOI.value:
            filter_format = FilterFormat.HWOI
        else:
            mace_check(False, "filter format %d not supported" %
                       filter_format_value)

        return filter_format

    def set_filter_format(self, filter_format):
        arg = ConverterUtil.get_arg(self._model,
                                    MaceKeyword.mace_filter_format_str)
        arg.i = filter_format.value

    def construct_ops_and_consumers(self):
        self._ops.clear()
        self._consumers.clear()
        self._producer.clear()
        for op in self._model.op:
            self._ops[op.name] = op
        for tensor in self._model.tensors:
            self._consts[tensor.name] = tensor
        for op in self._ops.values():
            for input_tensor in op.input:
                if input_tensor not in self._consumers:
                    self._consumers[input_tensor] = []
                self._consumers[input_tensor].append(op)

            for output_tensor in op.output:
                self._producer[output_tensor] = op
        for input_node in self._option.input_nodes.values():
            input_node_existed = False
            for op in self._model.op:
                if input_node.name in op.output:
                    input_node_existed = True
                    break
            if not input_node_existed:
                op = mace_pb2.OperatorDef()
                op.name = self.normalize_op_name(input_node.name)
                op.type = 'Input'
                op.output.extend([input_node.name])
                output_shape = op.output_shape.add()
                output_shape.dims.extend(input_node.shape)
                if ConverterUtil.data_format(
                        self._consumers[input_node.name][0]) \
                        == DataFormat.NCHW:
                    self.transpose_shape(output_shape.dims, [0, 3, 1, 2])
                    ConverterUtil.add_data_format_arg(op, DataFormat.NCHW)
                else:
                    ConverterUtil.add_data_format_arg(op, DataFormat.NHWC)
                self._producer[op.output[0]] = op

    @staticmethod
    def replace(obj_list, source, target):
        for i in xrange(len(obj_list)):
            if obj_list[i] == source:
                obj_list[i] = target

    @staticmethod
    def transpose_shape(shape, order):
        transposed_shape = []
        for i in xrange(len(order)):
            transposed_shape.append(shape[order[i]])
        shape[:] = transposed_shape[:]

    @staticmethod
    def normalize_op_name(name):
        return name.replace(':', '_')

    def get_tensor_shape(self, tensor):
        producer = self._producer[tensor]
        for i in xrange(len(producer.output)):
            if producer.output[i] == tensor:
                return list(producer.output_shape[i].dims)

    def consumer_count(self, tensor_name):
        return len(self._consumers.get(tensor_name, []))

    def is_op_output_node(self, op):
        output_node_tensor_names = [out for out in
                                    self._option.output_nodes]
        for output in op.output:
            if output in output_node_tensor_names:
                return True

        return False

    def safe_remove_node(self, op, replace_op, remove_input_tensor=False):
        """remove op.
        1. change the inputs of its consumers to the outputs of replace_op
        2. if the op is output node, change output node to replace op"""

        if replace_op is None:
            # When no replace op specified, we change the inputs of
            # its consumers to the input of the op. This handles the case
            # that the op is identity op and its input is a tensor.
            mace_check(len(op.output) == 1 and len(op.input) == 1,
                       "cannot remove op that w/o replace op specified"
                       " and input/output length > 1" + str(op))

            for consumer_op in self._consumers.get(op.output[0], []):
                self.replace(consumer_op.input, op.output[0], op.input[0])

            mace_check(op.output[0] not in self._option.output_nodes,
                       "cannot remove op that is output node")
        else:
            mace_check(len(op.output) == len(replace_op.output),
                       "cannot remove op since len(op.output) "
                       "!= len(replace_op.output)")

            for i in xrange(len(op.output)):
                for consumer_op in self._consumers.get(op.output[i], []):
                    self.replace(consumer_op.input,
                                 op.output[i],
                                 replace_op.output[i])

            # if the op is output node, change replace_op output name to the op
            # output name
            for i in xrange(len(op.output)):
                if op.output[i] in self._option.output_nodes:
                    for consumer in self._consumers.get(
                            replace_op.output[i], []):
                        self.replace(consumer.input,
                                     replace_op.output[i],
                                     op.output[i])
                    replace_op.output[i] = op.output[i]

        if remove_input_tensor:
            for input_name in op.input:
                if input_name in self._consts:
                    const_tensor = self._consts[input_name]
                    self._model.tensors.remove(const_tensor)

        self._model.op.remove(op)

    def add_in_out_tensor_info(self):
        net = self._model
        for input_node in self._option.input_nodes.values():
            input_info = net.input_info.add()
            input_info.name = input_node.name
            input_info.dims.extend(input_node.shape)

        for output_node in self._option.output_nodes.values():
            output_info = net.output_info.add()
            output_info.name = output_node.name
            output_info.dims.extend(
                self._producer[output_node.name].output_shape[0].dims)

        return False

    def remove_identity_op(self):
        net = self._model
        for op in net.op:
            if op.type == 'Identity':
                print("Remove identity: %s(%s)" % (op.name, op.type))
                self.safe_remove_node(op,
                                      self._producer.get(op.input[0], None))
                return True

        return False

    def transform_global_pooling(self):
        net = self._model
        for op in net.op:
            if op.type == MaceOp.Pooling.name and \
                            ConverterUtil.get_arg(op,
                                                  MaceKeyword.mace_global_pooling_str) is not None:  # noqa
                print("Transform global pooling: %s(%s)" % (op.name, op.type))
                input_shape = self._producer[op.input[0]].output_shape[0].dims
                if ConverterUtil.data_format(op) == DataFormat.NHWC:
                    kernel_shape = input_shape[1:3]
                else:
                    kernel_shape = input_shape[2:4]
                ConverterUtil.get_arg(op,
                                      MaceKeyword.mace_kernel_str).ints[:] \
                    = kernel_shape[:]

        return False

    def fold_batchnorm(self):
        net = self._model
        for op in net.op:
            if (op.type == MaceOp.Eltwise.name
                    and ConverterUtil.get_arg(
                        op, MaceKeyword.mace_element_type_str).i
                    == EltwiseType.PROD.value) \
                    and len(op.input) == 2 \
                    and op.input[1] in self._consts \
                    and self.consumer_count(op.output[0]) == 1 \
                    and not self.is_op_output_node(op):
                consumer_op = self._consumers[op.output[0]][0]
                if (consumer_op.type == MaceOp.Eltwise.name
                    and ConverterUtil.get_arg(
                        op, MaceKeyword.mace_element_type_str).i
                        == EltwiseType.SUM.value
                    or consumer_op.type == MaceOp.BiasAdd.name) \
                        and len(consumer_op.input) == 2 \
                        and consumer_op.input[1] in self._consts \
                        and len(self._consts[consumer_op.input[1]].dims) == 1:
                    print("Fold batchnorm: %s(%s)" % (op.name, op.type))
                    consumer_op.type = MaceOp.FoldedBatchNorm.name
                    consumer_op.input[:] = [op.input[0], op.input[1],
                                            consumer_op.input[1]]

                    self.safe_remove_node(op, None)
                    return True

        return False

    def fold_conv_and_bn(self):
        net = self._model
        for op in net.op:
            if (op.type == MaceOp.Conv2D.name
                or op.type == MaceOp.Deconv2D.name) \
                    and self.consumer_count(op.output[0]) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                if consumer_op.type == MaceOp.FoldedBatchNorm.name:
                    print("Fold conv and bn: %s(%s)" % (op.name, op.type))
                    filter = self._consts[op.input[1]]
                    scale = self._consts[consumer_op.input[1]]
                    idx = 0
                    filter_format = self.filter_format()
                    if filter_format == FilterFormat.HWIO:
                        for hwi in xrange(filter.dims[0] * filter.dims[1]
                                          * filter.dims[2]):
                            for o in xrange(filter.dims[3]):
                                filter.float_data[idx] *= scale.float_data[o]
                                idx += 1
                    elif filter_format == FilterFormat.OIHW:
                        for o in xrange(filter.dims[0]):
                            for hwi in xrange(filter.dims[1] * filter.dims[2]
                                              * filter.dims[3]):
                                filter.float_data[idx] *= scale.float_data[o]
                                idx += 1
                    else:
                        mace_check(False, "filter format %s not supported" %
                                   filter_format)

                    # change BN to BiasAdd
                    consumer_op.type = MaceOp.BiasAdd.name
                    del consumer_op.input[1]

                    # remove scale tensor
                    net.tensors.remove(scale)
                    return True

        return False

    def fold_depthwise_conv_and_bn(self):
        net = self._model
        for op in net.op:
            if op.type == MaceOp.DepthwiseConv2d.name \
                    and self.consumer_count(op.output[0]) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                if consumer_op.type == MaceOp.FoldedBatchNorm.name:
                    print("Fold depthwise conv and bn: %s(%s)"
                          % (op.name, op.type))
                    filter = self._consts[op.input[1]]
                    scale = self._consts[consumer_op.input[1]]
                    idx = 0

                    filter_format = self.filter_format()
                    if filter_format == FilterFormat.HWIO:
                        for hw in xrange(filter.dims[0] * filter.dims[1]):
                            for i in xrange(filter.dims[2]):
                                for o in xrange(filter.dims[3]):
                                    filter.float_data[idx] *= scale.float_data[
                                        i * filter.dims[3] + o]
                                    idx += 1
                    elif filter_format == FilterFormat.OIHW:
                        for o in xrange(filter.dims[0]):
                            for i in xrange(filter.dims[1]):
                                for hw in xrange(filter.dims[2]
                                                 * filter.dims[3]):
                                    filter.float_data[idx] *= scale.float_data[
                                        i * filter.dims[0] + o]
                                    idx += 1
                    else:
                        mace_check(False, "filter format %s not supported" %
                                   filter_format)

                    # change BN to BiasAdd
                    consumer_op.type = MaceOp.BiasAdd.name
                    del consumer_op.input[1]

                    # remove scale tensor
                    net.tensors.remove(scale)
                    return True

        return False

    @staticmethod
    def sort_feature_map_shape(shape, data_format):
        """Return shape in NHWC order"""
        batch = shape[0]
        if data_format == DataFormat.NHWC:
            height = shape[1]
            width = shape[2]
            channels = shape[3]
        else:
            height = shape[2]
            width = shape[3]
            channels = shape[1]
        return batch, height, width, channels

    @staticmethod
    def sort_filter_shape(filter_shape, filter_format):
        """Return filter shape in HWIO order"""
        if filter_format == FilterFormat.HWIO:
            filter_height = filter_shape[0]
            filter_width = filter_shape[1]
            in_channels = filter_shape[2]
            out_channels = filter_shape[3]
        elif filter_format == FilterFormat.OIHW:
            filter_height = filter_shape[2]
            filter_width = filter_shape[3]
            in_channels = filter_shape[1]
            out_channels = filter_shape[0]
        elif filter_format == FilterFormat.HWOI:
            filter_height = filter_shape[0]
            filter_width = filter_shape[1]
            in_channels = filter_shape[3]
            out_channels = filter_shape[2]
        else:
            mace_check(False, "filter format %s not supported" % filter_format)
        return filter_height, filter_width, in_channels, out_channels

    def check_if_gpu_use_winograd_conv(self, op):
        if not self._option.winograd:
            return False
        if op.type != MaceOp.Conv2D.name:
            return False

        filter_shape = self._consts[op.input[1]].dims
        output_shape = op.output_shape[0].dims
        strides = ConverterUtil.get_arg(op, MaceKeyword.mace_strides_str).ints
        dilations_arg = ConverterUtil.get_arg(op,
                                              MaceKeyword.mace_dilations_str)
        if dilations_arg is None:
            dilations = [1, 1]
        else:
            dilations = dilations_arg.ints
        filter_height, filter_width, in_channels, out_channels = \
            Transformer.sort_filter_shape(filter_shape, self.filter_format())
        batch, out_height, out_width, _ = Transformer.sort_feature_map_shape(
            output_shape, ConverterUtil.data_format(op))

        if filter_height != 3 or filter_width != 3 or strides[0] > 1 \
                or strides[1] > 1 or dilations[0] > 1 or dilations[1] > 1:
            return False
        self._gpu_wino_blk = self._option.winograd
        block_size = self._option.winograd
        blk_sqr = (block_size + 2) * (block_size + 2)
        width =\
            batch * ((out_height + block_size - 1) / block_size) *\
            ((out_width + block_size - 1) / block_size)
        if blk_sqr * in_channels >= OPENCL_IMAGE_MAX_SIZE \
                or blk_sqr * out_channels >= OPENCL_IMAGE_MAX_SIZE \
                or width >= OPENCL_IMAGE_MAX_SIZE:
            self._gpu_wino_blk = 2
            block_size = self._gpu_wino_blk
            blk_sqr = (block_size + 2) * (block_size + 2)
            width = \
                batch * ((out_height + block_size - 1) / block_size) * \
                ((out_width + block_size - 1) / block_size)
        return (blk_sqr * in_channels < OPENCL_IMAGE_MAX_SIZE) and \
               (blk_sqr * out_channels < OPENCL_IMAGE_MAX_SIZE) and \
               (width < OPENCL_IMAGE_MAX_SIZE)

    def transform_gpu_winograd(self):
        """Only gpu needs winograd transform."""
        net = self._model
        filter_format = self.filter_format()
        if self._option.device == DeviceType.GPU.value:
            for op in net.op:
                if op.type == MaceOp.Conv2D.name \
                        and self.check_if_gpu_use_winograd_conv(op):
                    print("Transform gpu winograd %s(%s)" % (op.name, op.type))
                    block_size = self._gpu_wino_blk
                    blk_sqr = (block_size + 2) * (block_size + 2)
                    output_shape = op.output_shape[0].dims
                    filter = self._consts[op.input[1]]
                    filter_shape = filter.dims
                    data_format = ConverterUtil.data_format(op)
                    filter_height, filter_width, in_channels, out_channels = \
                        Transformer.sort_filter_shape(filter_shape,
                                                      filter_format)
                    batch, out_height, out_width, _ = \
                        Transformer.sort_feature_map_shape(output_shape,
                                                           data_format)

                    # Input transform
                    wt_op = net.op.add()
                    wt_op.name = op.name + '_input_transform'
                    wt_op.type = MaceOp.WinogradTransform.name
                    wt_op.input.extend([op.input[0]])
                    wt_op.output.extend([wt_op.name])
                    wt_output_shape = wt_op.output_shape.add()
                    wt_output_width =\
                        batch * ((out_height + block_size - 1) / block_size) *\
                        ((out_width + block_size - 1) / block_size)
                    wt_output_shape.dims.extend(
                        [blk_sqr, in_channels, wt_output_width])

                    blk_size_arg = wt_op.arg.add()
                    blk_size_arg.name = MaceKeyword.mace_wino_block_size
                    blk_size_arg.i = block_size

                    if ConverterUtil.get_arg(op,
                                             MaceKeyword.mace_padding_str) \
                            is not None:
                        padding_arg = wt_op.arg.add()
                        padding_arg.name = MaceKeyword.mace_padding_str
                        padding_arg.i = ConverterUtil.get_arg(
                            op, MaceKeyword.mace_padding_str).i
                    elif ConverterUtil.get_arg(
                            op, MaceKeyword.mace_padding_values_str) \
                            is not None:
                        padding_arg = wt_op.arg.add()
                        padding_arg.name = MaceKeyword.mace_padding_values_str
                        padding_arg.ints.extend(ConverterUtil.get_arg(
                            op, MaceKeyword.mace_padding_values_str).ints)

                    # MatMul
                    matmul_op = net.op.add()
                    matmul_op.name = op.name + '_matmul'
                    matmul_op.type = MaceOp.MatMul.name
                    matmul_op.input.extend([op.input[1], wt_op.output[0]])
                    matmul_op.output.extend([matmul_op.name])
                    matmul_output_shape = matmul_op.output_shape.add()
                    matmul_output_shape.dims.extend(
                        [blk_sqr, out_channels, wt_output_width])

                    arg = matmul_op.arg.add()
                    arg.name = MaceKeyword.mace_winograd_filter_transformed
                    arg.i = 1

                    # Inverse transform
                    iwt_op = net.op.add()
                    iwt_op.name = op.name + '_inverse_transform'
                    iwt_op.type = MaceOp.WinogradInverseTransform.name
                    iwt_op.input.extend([matmul_op.output[0]])
                    # biasadd
                    if len(op.input) >= 3:
                        iwt_op.input.extend([op.input[2]])
                    iwt_op.output.extend(op.output)
                    iwt_output_shape = iwt_op.output_shape.add()
                    iwt_output_shape.dims.extend(op.output_shape[0].dims)

                    batch_arg = iwt_op.arg.add()
                    batch_arg.name = 'batch'
                    batch_arg.i = batch
                    height_arg = iwt_op.arg.add()
                    height_arg.name = 'height'
                    height_arg.i = out_height
                    width_arg = iwt_op.arg.add()
                    width_arg.name = 'width'
                    width_arg.i = out_width
                    blk_size_arg = iwt_op.arg.add()
                    blk_size_arg.name = MaceKeyword.mace_wino_block_size
                    blk_size_arg.i = block_size
                    ConverterUtil.add_data_format_arg(iwt_op, data_format)

                    self.safe_remove_node(op, iwt_op)

        return False

    def transform_add_to_biasadd(self):
        net = self._model
        for op in net.op:
            if op.type == 'Add' \
                    and len(op.input) == 2 \
                    and op.input[1] in self._consts \
                    and len(self._consts[op.input[1]].dims) == 1:
                print("Transform add to biasadd: %s(%s)" % (op.name, op.type))
                op.type = MaceOp.BiasAdd.name
                return True

        return False

    def fold_biasadd(self):
        net = self._model
        for op in net.op:
            if (((op.type == MaceOp.Conv2D.name
                  or op.type == MaceOp.Deconv2D.name
                  or op.type == MaceOp.DepthwiseConv2d.name
                  or op.type == MaceOp.FullyConnected.name)
                 and len(op.input) == 2)
                or (op.type == MaceOp.WinogradInverseTransform.name
                    and len(op.input) == 1)) \
                    and len(self._consumers.get(op.output[0], [])) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                if consumer_op.type == MaceOp.BiasAdd.name:
                    print("Fold biasadd: %s(%s)" % (op.name, op.type))
                    op.input.append(consumer_op.input[1])
                    self.safe_remove_node(consumer_op, op)
                    return True

        return False

    def flatten_atrous_conv(self):
        if self._option.device != DeviceType.GPU.value:
            return

        net = self._model
        for op in net.op:
            if (op.type == MaceOp.SpaceToBatchND.name
                    and len(self._consumers.get(op.output[0], [])) == 1):
                conv_op = self._consumers.get(op.output[0])[0]
                if (conv_op.type == MaceOp.Conv2D.name
                        or conv_op.type == MaceOp.DepthwiseConv2d.name) \
                        and len(self._consumers.get(conv_op.output[0], [])) == 1:  # noqa
                    b2s_op = self._consumers.get(conv_op.output[0])[0]
                    if b2s_op.type == MaceOp.BatchToSpaceND.name:
                        print "Flatten atrous convolution"
                        # Add args.
                        padding_arg_values = ConverterUtil.get_arg(
                            op,
                            MaceKeyword.mace_paddings_str).ints
                        blocks_arg_values = ConverterUtil.get_arg(
                            b2s_op,
                            MaceKeyword.mace_space_batch_block_shape_str).ints
                        dilation_arg = ConverterUtil.get_arg(
                            conv_op,
                            MaceKeyword.mace_dilations_str)
                        if dilation_arg is None:
                            dilation_arg = conv_op.arg.add()
                        dilation_arg.name = MaceKeyword.mace_dilations_str
                        dilation_arg.ints[:] = blocks_arg_values

                        padding_arg = ConverterUtil.get_arg(
                            conv_op,
                            MaceKeyword.mace_padding_str)
                        if padding_arg is None:
                            padding_arg = conv_op.arg.add()
                        padding_arg.name = MaceKeyword.mace_padding_str
                        if len(padding_arg_values) > 0 \
                                and padding_arg_values[0] > 0:
                            padding_arg.i = PaddingMode.SAME.value
                        else:
                            padding_arg.i = PaddingMode.VALID.value

                        strides_arg = ConverterUtil.get_arg(
                            conv_op,
                            MaceKeyword.mace_strides_str)
                        if strides_arg is None:
                            strides_arg = conv_op.arg.add()
                        strides_arg.name = MaceKeyword.mace_strides_str
                        strides_arg.ints[:] = [1, 1]

                        # update output shape
                        conv_op.output_shape[0].dims[:] = \
                            b2s_op.output_shape[0].dims[:]

                        self.safe_remove_node(op, None)
                        self.safe_remove_node(b2s_op, conv_op)
                        return True
        return False

    def fold_activation(self):
        net = self._model
        for op in net.op:
            if (op.type == MaceOp.Conv2D.name
                or op.type == MaceOp.Deconv2D.name
                or op.type == MaceOp.DepthwiseConv2d.name
                or op.type == MaceOp.FullyConnected.name
                or op.type == MaceOp.FoldedBatchNorm.name
                or op.type == MaceOp.WinogradInverseTransform.name) \
                    and len(self._consumers.get(op.output[0], [])) == 1:
                consumer_op = self._consumers[op.output[0]][0]
                if consumer_op.type == MaceOp.Activation.name \
                        and ConverterUtil.get_arg(
                            consumer_op,
                            MaceKeyword.mace_activation_type_str).s != 'PRELU':
                    print("Fold activation: %s(%s)" % (op.name, op.type))
                    op.name = consumer_op.name
                    op.output[0] = consumer_op.output[0]
                    for arg in consumer_op.arg:
                        if arg.name == MaceKeyword.mace_activation_type_str \
                                or arg.name == MaceKeyword.mace_activation_max_limit_str:  # noqa
                            op.arg.extend([arg])

                    self.safe_remove_node(consumer_op, op)
                    return True

        return False

    def transpose_data_format(self):
        net = self._model

        for op in net.op:
            # transpose args
            if op.type == MaceOp.Pad.name:
                for arg in op.arg:
                    if arg.name == MaceKeyword.mace_paddings_str:
                        mace_check(len(arg.ints) == 8,
                                   "pad dim rank should be 8.")
                        if ConverterUtil.data_format(op) == DataFormat.NHWC \
                                and self._target_data_format == DataFormat.NCHW:  # noqa
                            print("Transpose pad args: %s(%s)"
                                  % (op.name, op.type))
                            self.transpose_shape(arg.ints,
                                                 [0, 1, 6, 7, 2, 3, 4, 5])
                        elif ConverterUtil.data_format(op) == DataFormat.NCHW \
                                and self._target_data_format == DataFormat.NHWC:  # noqa
                            print("Transpose pad args: %s(%s)"
                                  % (op.name, op.type))
                            self.transpose_shape(arg.ints,
                                                 [0, 1, 4, 5, 6, 7, 2, 3])
            elif op.type == MaceOp.Concat.name or op.type == MaceOp.Slice.name:
                for arg in op.arg:
                    if arg.name == MaceKeyword.mace_axis_str:
                        if ConverterUtil.data_format(op) == DataFormat.NHWC \
                                and self._target_data_format == DataFormat.NCHW:  # noqa
                            print("Transpose slice args: %s(%s)"
                                  % (op.name, op.type))
                            mace_check(arg.i == 3,
                                       'only support concat at '
                                       'channel dimension')
                            arg.i = 1
                        elif ConverterUtil.data_format(op) == DataFormat.NCHW \
                                and self._target_data_format == DataFormat.NHWC:  # noqa
                            print("Transpose slice args: %s(%s)"
                                  % (op.name, op.type))
                            mace_check(arg.i == 1,
                                       "only support concat at "
                                       "channel dimension")
                            arg.i = 3

            elif op.type == MaceOp.Squeeze.name:
                for arg in op.arg:
                    if arg.name == MaceKeyword.mace_axis_str:
                        if ConverterUtil.data_format(
                                op) == DataFormat.NHWC \
                                and self._target_data_format == DataFormat.NCHW:  # noqa
                            print("Transpose squeeze args: %s(%s)"
                                  % (op.name, op.type))
                            mace_check(list(arg.ints) == [1, 2],
                                       'only support squeeze at at [1, 2]')
                            arg.ints[:] = [2, 3]
                        elif ConverterUtil.data_format(
                                op) == DataFormat.NCHW \
                                and self._target_data_format == DataFormat.NHWC:  # noqa
                            print("Transpose squeeze args: %s(%s)"
                                  % (op.name, op.type))
                            mace_check(list(arg.ints) == [2, 3],
                                       'only support squeeze at at [2, 3]')
                            arg.ints[:] = [1, 2]

            elif op.type == MaceOp.ReduceMean.name:
                for arg in op.arg:
                    if arg.name == MaceKeyword.mace_axis_str:
                        if ConverterUtil.data_format(
                                op) == DataFormat.NHWC \
                                and self._target_data_format == DataFormat.NCHW:  # noqa
                            print("Transpose reduce mean args: %s(%s)"
                                  % (op.name, op.type))
                            reduce_axises = list(arg.ints)
                            new_axises = []
                            for i in range(len(reduce_axises)):
                                idx = reduce_axises[i]
                                if idx == 1 or idx == 2:
                                    new_axises.append(idx + 1)
                                elif idx == 3:
                                    new_axises.append(1)
                                else:
                                    new_axises.append(idx)
                            new_axises.sort()
                            arg.ints[:] = []
                            arg.ints.extend(new_axises)
                        elif ConverterUtil.data_format(
                                op) == DataFormat.NCHW \
                                and self._target_data_format == DataFormat.NHWC:  # noqa
                            print("Transpose reduce mean args: %s(%s)"
                                  % (op.name, op.type))
                            reduce_axises = list(arg.ints)
                            new_axises = []
                            for i in range(len(reduce_axises)):
                                idx = reduce_axises[i]
                                if idx == 2 or idx == 3:
                                    new_axises.append(idx - 1)
                                elif idx == 1:
                                    new_axises.append(3)
                                else:
                                    new_axises.append(idx)
                            new_axises.sort()
                            arg.ints[:] = []
                            arg.ints.extend(new_axises)

            # transpose op output shape
            data_format = ConverterUtil.data_format(op)
            if data_format is not None \
                    and data_format != self._target_data_format:
                print("Transpose output shapes: %s(%s)" % (op.name, op.type))
                if self._target_data_format == DataFormat.NHWC:  # NCHW -> NHWC
                    for output_shape in op.output_shape:
                        if len(output_shape.dims) == 4:
                            self.transpose_shape(output_shape.dims,
                                                 [0, 2, 3, 1])
                else:  # NHWC -> NCHW
                    for output_shape in op.output_shape:
                        if len(output_shape.dims) == 4:
                            self.transpose_shape(output_shape.dims,
                                                 [0, 3, 1, 2])
                ConverterUtil.get_arg(op,
                                      MaceKeyword.mace_data_format_str).i = \
                    self._target_data_format.value

        # transpose input/output
        if self._target_data_format == DataFormat.NCHW:
            print("Transpose input/output to NCHW")
            for input_node in self._option.input_nodes.values():
                new_input_name = MaceKeyword.mace_input_node_name \
                                 + '_' + input_node.name
                op = net.op.add()
                op.name = self.normalize_op_name(input_node.name)
                op.input.extend([new_input_name])
                op.output.extend([input_node.name])
                output_shape = op.output_shape.add()
                output_shape.dims.extend(input_node.shape)
                if len(output_shape.dims) == 4:
                    op.type = MaceOp.Transpose.name
                    self.transpose_shape(output_shape.dims, [0, 3, 1, 2])

                    dims_arg = op.arg.add()
                    dims_arg.name = MaceKeyword.mace_dims_str
                    dims_arg.ints.extend([0, 3, 1, 2])
                else:
                    op.type = MaceOp.Identity.name

                ConverterUtil.add_data_format_arg(op, DataFormat.NCHW)

            for output_node in self._option.output_nodes.values():
                output_name = MaceKeyword.mace_output_node_name \
                              + '_' + output_node.name
                op = self._model.op.add()
                op.name = self.normalize_op_name(output_name)
                op.input.extend([output_node.name])
                op.output.extend([output_name])
                output_shape = op.output_shape.add()
                output_shape.dims.extend(
                    self._producer[output_node.name].output_shape[0].dims)
                if len(output_shape.dims) == 4:
                    op.type = MaceOp.Transpose.name
                    self.transpose_shape(output_shape.dims, [0, 2, 3, 1])

                    dims_arg = op.arg.add()
                    dims_arg.name = MaceKeyword.mace_dims_str
                    dims_arg.ints.extend([0, 2, 3, 1])

                    ConverterUtil.add_data_format_arg(op, DataFormat.NHWC)
                else:
                    op.type = MaceOp.Identity.name

            self._input_output_added = True

        return False

    def transpose_filters(self):
        net = self._model
        filter_format = self.filter_format()

        print("Transpose filters to OIHW")
        # transpose filter to OIHW/MIHW for tensorflow (HWIO/HWIM)
        if filter_format == FilterFormat.HWIO:
            for op in net.op:
                if op.type == MaceOp.Conv2D.name \
                        or op.type == MaceOp.Deconv2D.name \
                        or op.type == MaceOp.DepthwiseConv2d.name:
                    filter = self._consts[op.input[1]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(3, 2, 0, 1)
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                if (op.type == MaceOp.MatMul.name and
                            ConverterUtil.get_arg(op, MaceKeyword.mace_winograd_filter_transformed) is not None):  # noqa
                    filter = self._consts[op.input[0]]
                    filter_data = np.array(filter.float_data).reshape(
                        filter.dims)
                    filter_data = filter_data.transpose(3, 2, 0, 1)
                    filter.float_data[:] = filter_data.flat
                    filter.dims[:] = filter_data.shape
                if op.type == MaceOp.FullyConnected.name:
                    weight = self._consts[op.input[1]]
                    if len(weight.dims) == 4:
                        weight_data = np.array(weight.float_data).reshape(
                            weight.dims)
                        weight_data = weight_data.transpose(3, 2, 0, 1)
                        weight.float_data[:] = weight_data.flat
                        weight.dims[:] = weight_data.shape

            self.set_filter_format(FilterFormat.OIHW)
        for op in net.op:
            if op.type == MaceOp.Deconv2D.name:
                filter = self._consts[op.input[1]]
                filter_data = np.array(filter.float_data).reshape(
                    filter.dims)
                filter_data = filter_data.transpose(1, 0, 2, 3)
                filter.float_data[:] = filter_data.flat
                filter.dims[:] = filter_data.shape

        return False

    def reshape_fc_weight(self):
        print("Reshape fully connected weight shape")
        net = self._model
        for op in net.op:
            if op.type == MaceOp.FullyConnected.name:
                weight = self._consts[op.input[1]]
                if len(weight.dims) == 2:
                    input_op = self._producer[op.input[0]]
                    input_shape = list(input_op.output_shape[0].dims)
                    input_data_format = ConverterUtil.data_format(input_op)
                    weight.dims[:] = [weight.dims[0]] + input_shape[1:]
                    if input_data_format == DataFormat.NHWC:
                        self.transpose_shape(weight.dims, [0, 3, 1, 2])

        return False

    def buffer_to_image(self, op, input_idx, input_type):
        net = self._model
        input_name = op.input[input_idx]
        op_def = net.op.add()
        op_def.name = input_name.replace(':', '_') + "_b2i"
        output_name = op_def.name
        op_def.type = MaceKeyword.mace_buffer_to_image
        op_def.input.extend([input_name])
        op_def.output.extend([output_name])

        arg = op_def.arg.add()
        arg.name = MaceKeyword.mace_buffer_type
        arg.i = input_type.value
        arg = op_def.arg.add()
        arg.name = MaceKeyword.mace_mode
        arg.i = 0

        if input_type == OpenCLBufferType.WINOGRAD_FILTER:
            blk_sqr = op.output_shape[0].dims[0]
            wino_blk = int(np.sqrt(blk_sqr)) - 2
            wino_arg = op_def.arg.add()
            wino_arg.name = MaceKeyword.mace_wino_block_size
            wino_arg.i = wino_blk

        op.input[input_idx] = output_name

    def transform_buffer_image(self):
        if self._option.device != DeviceType.GPU.value:
            return False

        print("Transform buffer to image")

        net = self._model
        for op in net.op:
            if op.type == MaceOp.Conv2D.name \
                    or op.type == MaceOp.Deconv2D.name:
                self.buffer_to_image(op, 1, OpenCLBufferType.CONV2D_FILTER)
                if len(op.input) >= 3:
                    self.buffer_to_image(op, 2, OpenCLBufferType.ARGUMENT)
            elif op.type == MaceOp.DepthwiseConv2d.name:
                self.buffer_to_image(op, 1, OpenCLBufferType.DW_CONV2D_FILTER)
                if len(op.input) >= 3:
                    self.buffer_to_image(op, 2, OpenCLBufferType.ARGUMENT)
            elif op.type == MaceOp.BiasAdd.name:
                self.buffer_to_image(op, 1, OpenCLBufferType.ARGUMENT)
            elif op.type == MaceOp.Eltwise.name and len(op.input) == 2:
                if op.input[0] in self._consts \
                        and len(self._consts[op.input[0]].dims) == 1:
                    self.buffer_to_image(op, 0, OpenCLBufferType.ARGUMENT)
                if op.input[1] in self._consts \
                        and len(self._consts[op.input[1]].dims) == 1:
                    self.buffer_to_image(op, 1, OpenCLBufferType.ARGUMENT)
            elif op.type == MaceOp.FoldedBatchNorm.name:
                self.buffer_to_image(op, 1, OpenCLBufferType.ARGUMENT)
                self.buffer_to_image(op, 2, OpenCLBufferType.ARGUMENT)
                if len(op.input) >= 4:
                    self.buffer_to_image(op, 3, OpenCLBufferType.ARGUMENT)
            elif op.type == MaceOp.MatMul.name and \
                    ConverterUtil.get_arg(op,
                                          MaceKeyword.mace_winograd_filter_transformed) is not None:  # noqa
                self.buffer_to_image(op, 0, OpenCLBufferType.WINOGRAD_FILTER)
            elif op.type == MaceOp.WinogradInverseTransform.name \
                    and len(op.input) >= 2:
                self.buffer_to_image(op, 1, OpenCLBufferType.ARGUMENT)
            elif op.type == MaceOp.FullyConnected.name:
                self.buffer_to_image(op, 1, OpenCLBufferType.WEIGHT_WIDTH)
                if len(op.input) >= 3:
                    self.buffer_to_image(op, 2, OpenCLBufferType.ARGUMENT)
            elif op.type == MaceOp.Activation.name:
                if ConverterUtil.get_arg(op,
                                         MaceKeyword.mace_activation_type_str).s == ActivationType.PRELU.name:  # noqa
                    self.buffer_to_image(op, 1, OpenCLBufferType.ARGUMENT)

        for input_node in self._option.input_nodes.values():
            new_input_name = MaceKeyword.mace_input_node_name \
                             + '_' + input_node.name
            op_def = self._model.op.add()

            op_def.name = self.normalize_op_name(input_node.name)
            op_def.type = MaceKeyword.mace_buffer_to_image
            op_def.input.extend([new_input_name])
            op_def.output.extend([input_node.name])
            output_shape = op_def.output_shape.add()
            output_shape.dims.extend(input_node.shape)

            arg = op_def.arg.add()
            arg.name = MaceKeyword.mace_buffer_type
            arg.i = OpenCLBufferType.IN_OUT_CHANNEL.value

            ConverterUtil.add_data_format_arg(op_def, DataFormat.NHWC)

        for output_node in self._option.output_nodes.values():
            output_name = MaceKeyword.mace_output_node_name \
                          + '_' + output_node.name
            op_def = self._model.op.add()
            op_def.name = self.normalize_op_name(output_name)
            op_def.type = MaceKeyword.mace_image_to_buffer
            op_def.input.extend([output_node.name])
            op_def.output.extend([output_name])
            if output_node.shape:
                output_shape = op_def.output_shape.add()
                output_shape.dims.extend(output_node.shape)

            arg = op_def.arg.add()
            arg.name = MaceKeyword.mace_buffer_type
            arg.i = OpenCLBufferType.IN_OUT_CHANNEL.value

            ConverterUtil.add_data_format_arg(op_def, DataFormat.NHWC)

        self._input_output_added = True

        return False

    def fold_reshape(self):
        net = self._model
        for op in net.op:
            if op.type == MaceOp.Softmax.name:
                # see if possible to fold
                # Reshape(xd->2d) + Softmax(2d) + Reshape(xd) to Softmax(xd)
                should_fold = False
                if op.input[0] in self._producer \
                        and self._producer[op.input[0]].type \
                        == MaceOp.Reshape.name \
                        and len(op.output_shape[0].dims) == 2 \
                        and self.consumer_count(op.output[0]) == 1:
                    producer = self._producer[op.input[0]]
                    consumer = self._consumers[op.output[0]][0]

                    if (consumer.type == MaceOp.Reshape.name
                        and op.output_shape[0].dims[-1]
                            == consumer.output_shape[0].dims[-1]
                        and op.output_shape[0].dims[-1] != -1
                        and self.get_tensor_shape(producer.input[0])
                            == consumer.output_shape[0].dims):
                            should_fold = True

                if should_fold:
                    print(
                        "Fold reshape and softmax: %s(%s)"
                        % (op.name, op.type))
                    producer = self._producer[op.input[0]]
                    consumer = self._consumers[op.output[0]][0]
                    op.output_shape[0].dims[:] = self.get_tensor_shape(
                        producer.input[0])

                    # if there is a shape op, remove it too
                    if (consumer.input[1] in self._producer
                        and self._producer[consumer.input[1]].type
                            == 'Shape'):
                        self.safe_remove_node(
                            self._producer[consumer.input[1]], None,
                            remove_input_tensor=True)
                    # remove consumer reshape
                    self.safe_remove_node(consumer, op,
                                          remove_input_tensor=True)
                    # remove producer reshape
                    self.safe_remove_node(producer,
                                          self._producer.get(producer.input[0],
                                                             None),
                                          remove_input_tensor=True)

                    return True
        return False

    def transform_matmul_to_fc(self):
        net = self._model
        filter_format = self.filter_format()
        for op in net.op:
            # transform reshape + matmul -> fc
            # work for TensorFlow
            if op.type == MaceOp.MatMul.name and \
                    filter_format == FilterFormat.HWIO:
                producer = self._producer[op.input[0]]
                weight = self._consts[op.input[1]]
                if len(weight.dims) == 2 \
                    and producer.type == MaceOp.Reshape.name \
                    and len(producer.output) == 1 \
                    and producer.input[1] in self._consts \
                        and len(producer.output_shape[0].dims) == 2:
                    input_op = self._producer[producer.input[0]]
                    input_shape = input_op.output_shape[0].dims
                    feature_size = np.prod(input_shape[1:])
                    self.safe_remove_node(producer, input_op,
                                          remove_input_tensor=True)
                    if feature_size == producer.output_shape[0].dims[1]:
                        print 'convert reshape and matmul to fc'
                        op.type = MaceOp.FullyConnected.name
                        weight_data = np.array(weight.float_data).reshape(
                            weight.dims)
                        weight.dims[:] = input_shape[1:] + \
                            [weight_data.shape[1]]
                        return True
                elif len(weight.dims) == 2 and \
                        len(producer.output_shape[0].dims) == 2 and \
                        weight.dims[0] == producer.output_shape[0].dims[1]:
                    print 'convert matmul to fc'
                    op.type = MaceOp.FullyConnected.name
                    weight_data = np.array(weight.float_data).reshape(
                        weight.dims)
                    weight.dims[:] = [1, 1] + list(weight_data.shape)
                    return True

        return False

    def transform_global_conv_to_fc(self):
        """Transform global conv to fc should be placed after transposing
        input/output and filter"""
        # if self._option.device == DeviceType.GPU.value:
        # return False

        net = self._model
        for op in net.op:
            if op.type == MaceOp.Conv2D.name:
                producer = self._producer[op.input[0]]
                input_shape = producer.output_shape[0].dims
                batch, height, width, channels = self.sort_feature_map_shape(
                    input_shape, ConverterUtil.data_format(producer))
                filter = self._consts[op.input[1]]
                filter_shape = filter.dims
                filter_height, filter_width, in_channels, out_channels = \
                    self.sort_filter_shape(filter_shape, self.filter_format())
                zero_padding = True
                padding_arg = ConverterUtil.get_arg(op,
                                                    MaceKeyword.mace_padding_str)  # noqa
                if padding_arg is not None:
                    if padding_arg.i != PaddingMode.VALID.value:
                        zero_padding = False
                else:
                    padding_value_arg = ConverterUtil.get_arg(op,
                                                              MaceKeyword.mace_padding_values_str)  # noqa
                    if padding_value_arg is not None:
                        if not all(v == 0 for v in padding_value_arg.ints):
                            zero_padding = False

                if height == filter_height and width == filter_width \
                        and zero_padding:
                    print("transform global conv to fc %s(%s)"
                          % (op.name, op.type))
                    op.type = MaceOp.FullyConnected.name

    def add_device(self):
        # TODO(liuqi) add device definition in OperatorDef
        net = self._model
        for op in net.op:
            arg = op.arg.add()
            arg.name = MaceKeyword.mace_device
            arg.i = self._option.device

        return False

    def update_float_op_data_type(self):
        print("update op with float data type")
        net = self._model
        for op in net.op:
            data_type_arg = ConverterUtil.get_arg(
                op, MaceKeyword.mace_op_data_type_str)
            if not data_type_arg:
                data_type_arg = op.arg.add()
                data_type_arg.name = MaceKeyword.mace_op_data_type_str
                data_type_arg.i = self._option.data_type
            elif data_type_arg.i != self._option.data_type \
                    and data_type_arg.i == mace_pb2.DT_FLOAT:
                data_type_arg.i = self._option.data_type

        return False

    def sort_dfs(self, op, visited, sorted_nodes):
        visited.update([op.name])
        if len(op.input) > 0:
            for input_tensor in op.input:
                producer_op = self._producer.get(input_tensor, None)
                if producer_op is None:
                    pass
                elif producer_op.name not in visited:
                    self.sort_dfs(producer_op, visited, sorted_nodes)
        sorted_nodes.append(op)

    def add_mace_input_and_output_nodes(self):
        if self._input_output_added:
            return

        print("add mace input and output nodes")

        for input_node in self._option.input_nodes.values():
            new_input_name = MaceKeyword.mace_input_node_name \
                             + '_' + input_node.name
            op_def = self._model.op.add()
            op_def.name = self.normalize_op_name(input_node.name)
            op_def.type = MaceOp.Identity.name
            op_def.input.extend([new_input_name])
            op_def.output.extend([input_node.name])
            output_shape = op_def.output_shape.add()
            output_shape.dims.extend(input_node.shape)

            ConverterUtil.add_data_format_arg(op_def, DataFormat.NHWC)

        for output_node in self._option.output_nodes.values():
            output_name = MaceKeyword.mace_output_node_name \
                          + '_' + output_node.name
            op_def = self._model.op.add()
            op_def.name = self.normalize_op_name(output_name)
            op_def.type = MaceOp.Identity.name
            op_def.input.extend([output_node.name])
            op_def.output.extend([output_name])
            output_shape = op_def.output_shape.add()
            output_shape.dims.extend(
                self._producer[output_node.name].output_shape[0].dims)

    def sort_by_execution(self):
        print("Sort by execution")
        net = self._model
        visited = set()
        sorted_nodes = []

        for output_node in self._option.output_nodes:
            output_tensor = MaceKeyword.mace_output_node_name \
                            + '_' + output_node
            mace_check(output_tensor in self._producer,
                       "output_tensor %s not existed in model" % output_tensor)
            self.sort_dfs(self._producer[output_tensor], visited, sorted_nodes)

        del net.op[:]
        net.op.extend(sorted_nodes)

        print("Final ops:")
        for op in net.op:
            print("%s (%s): %s" % (op.name, op.type, [
                out_shape.dims for out_shape in op.output_shape]))
        return False
