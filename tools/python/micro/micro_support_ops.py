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

from enum import Enum
from py_proto import mace_pb2
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp
from utils.config_parser import DataFormat
from utils.config_parser import ModelKeys
from utils.config_parser import Platform
from utils.util import mace_check

import copy


class OpDescriptor:
    def __init__(self, src_path, class_name, type,
                 data_type, data_format, tag=None):
        self.src_path = src_path
        self.class_name = class_name
        self.type = type
        self.data_type = data_type
        self.data_format = data_format
        self.tag = tag
        self.name = None
        self.idx = -1


McSupportedOps = [
    OpDescriptor('micro/ops/argmax.h', 'ArgMaxOp<mifloat>', MaceOp.ArgMax.name,
                 mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/nhwc/conv_2d_ref.h', 'Conv2dRefOp',
                 MaceOp.Conv2D.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC, None),
    OpDescriptor('micro/ops/nhwc/conv_2d_c4_s4.h', 'Conv2dC4S4Op',
                 MaceOp.Conv2D.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC, 'c4s4'),
    OpDescriptor('micro/ops/nhwc/conv_2d_c3_s4.h', 'Conv2dC3S4Op',
                 MaceOp.Conv2D.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC, 'c3s4'),
    OpDescriptor('micro/ops/nhwc/conv_2d_c2_s4.h', 'Conv2dC2S4Op',
                 MaceOp.Conv2D.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC, 'c2s4'),
    OpDescriptor('micro/ops/cast.h', 'CastOp',
                 MaceOp.Cast.name, mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/nhwc/pooling_ref.h', 'PoolingRefOp',
                 MaceOp.Pooling.name, mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/nhwc/pooling_s4.h', 'PoolingS4Op',
                 MaceOp.Pooling.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC, "s4"),
    OpDescriptor('micro/ops/squeeze.h', 'SqueezeOp', MaceOp.Squeeze.name,
                 mace_pb2.DT_FLOAT, None),
    OpDescriptor('micro/ops/softmax.h', 'SoftmaxOp', MaceOp.Softmax.name,
                 mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/eltwise.h', 'EltwiseOp<mifloat>',
                 MaceOp.Eltwise.name, mace_pb2.DT_FLOAT, None),
    OpDescriptor('micro/ops/eltwise.h', 'EltwiseOp<int32_t>',
                 MaceOp.Eltwise.name, mace_pb2.DT_INT32, None),
    OpDescriptor('micro/ops/activation.h', 'ActivationOp',
                 MaceOp.Activation.name, mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/strided_slice.h', 'StridedSliceOp<mifloat>',
                 MaceOp.StridedSlice.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC),
    OpDescriptor('micro/ops/strided_slice.h', 'StridedSliceOp<int32_t>',
                 MaceOp.StridedSlice.name, mace_pb2.DT_INT32,
                 DataFormat.NHWC),
    OpDescriptor('micro/ops/reduce.h', 'ReduceOp<mifloat>', MaceOp.Reduce.name,
                 mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/reduce.h', 'ReduceOp<int32_t>', MaceOp.Reduce.name,
                 mace_pb2.DT_INT32, DataFormat.NHWC),
    OpDescriptor('micro/ops/stack.h', 'StackOp<mifloat>', MaceOp.Stack.name,
                 mace_pb2.DT_FLOAT, None),
    OpDescriptor('micro/ops/stack.h', 'StackOp<int32_t>', MaceOp.Stack.name,
                 mace_pb2.DT_INT32, None),
    OpDescriptor('micro/ops/bias_add.h', 'BiasAddOp', MaceOp.BiasAdd.name,
                 mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/matmul.h', 'MatMulOp', MaceOp.MatMul.name,
                 mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/nhwc/batch_norm.h', 'BatchNormOp',
                 MaceOp.BatchNorm.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC),
    OpDescriptor('micro/ops/shape.h', 'ShapeOp', MaceOp.Shape.name,
                 mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/reshape.h', 'ReshapeOp', MaceOp.Reshape.name,
                 mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/expand_dims.h', 'ExpandDimsOp',
                 MaceOp.ExpandDims.name, mace_pb2.DT_FLOAT, DataFormat.NHWC),
    OpDescriptor('micro/ops/nhwc/depthwise_conv_2d_ref.h',
                 'DepthwiseConv2dRefOp',
                 MaceOp.DepthwiseConv2d.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC),
    OpDescriptor('micro/ops/nhwc/depthwise_conv_2d_kb4_s4.h',
                 'DepthwiseConv2dKB4S4Op',
                 MaceOp.DepthwiseConv2d.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC, 'kb4s4'),
    OpDescriptor('micro/ops/nhwc/depthwise_conv_2d_kb3_s4.h',
                 'DepthwiseConv2dKB3S4Op',
                 MaceOp.DepthwiseConv2d.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC, 'kb3s4'),
    OpDescriptor('micro/ops/nhwc/depthwise_conv_2d_kb2_s4.h',
                 'DepthwiseConv2dKB2S4Op',
                 MaceOp.DepthwiseConv2d.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC, 'kb2s4'),
    OpDescriptor('micro/ops/nhwc/depthwise_conv_2d_kb1_s4.h',
                 'DepthwiseConv2dKB1S4Op',
                 MaceOp.DepthwiseConv2d.name, mace_pb2.DT_FLOAT,
                 DataFormat.NHWC, 'kb1s4'),
]


class OpResolver:
    def __init__(self, pb_model, model_conf):
        self.net_def = pb_model
        self.op_desc_map = {}
        self.op_desc_list = []
        if model_conf[ModelKeys.platform] == Platform.TENSORFLOW:
            self.default_data_format = DataFormat.NHWC
        else:
            self.default_data_format = DataFormat.NCHW
        print("OpResolver set default_data_format: %s" %
              self.default_data_format)
        if ModelKeys.quantize in model_conf and \
                model_conf[ModelKeys.quantize] == 1:
            self.default_data_type = mace_pb2.DT_UINT8
        else:
            self.default_data_type = \
                model_conf.get(ModelKeys.data_type, mace_pb2.DT_FLOAT)

    def get_op_data_format(self, op_def):
        arg = self.get_op_def_arg(op_def, MaceKeyword.mace_data_format_str)
        if arg is None or arg.i == DataFormat.AUTO.value:
            return self.default_data_format
        else:
            return DataFormat(arg.i)

    def get_op_data_type(self, op_def):
        arg = self.get_op_def_arg(op_def, MaceKeyword.mace_op_data_type_str)
        if arg is None:
            return self.default_data_type
        else:
            return arg.i

    def get_op_def_arg(self, op_def, name):
        for arg in op_def.arg:
            if arg.name == name:
                return arg
        return None

    def get_op_def_input_dims(self, op_def, idx):
        input_name = op_def.input[idx]
        for const_tensor in self.net_def.tensors:
            if input_name == const_tensor.name:
                return const_tensor.dims
        for pre_op in self.net_def.op:
            for i in range(len(pre_op.output)):
                if input_name == pre_op.output[i]:
                    return pre_op.output_shape[i].dims
        return None

    def get_op_tag(self, op_def):
        if op_def.type == MaceOp.Conv2D.name:
            output_shape = op_def.output_shape[0].dims
            size = output_shape[0] * output_shape[1] * output_shape[2]
            if size >= 4:
                size = 4
            channel = output_shape[3]
            if channel >= 4:
                channel = 4
            if channel >= 2 and size >= 4:
                return ("c%ss%s" % (channel, size))
        elif op_def.type == MaceOp.DepthwiseConv2d.name:
            output_shape = op_def.output_shape[0].dims
            size = output_shape[0] * output_shape[1] * output_shape[2]
            if size >= 4:
                size = 4
            filter_dims = self.get_op_def_input_dims(op_def, 1)
            mace_check(filter_dims is not None, "Get filter dims failed.")
            k_batch = filter_dims[0]
            if k_batch >= 4:
                k_batch = 4
            if size >= 4:
                return ("kb%ss%s" % (k_batch, size))
        elif op_def.type == MaceOp.Pooling.name:
            kernels = self.get_op_def_arg(op_def, MaceKeyword.mace_kernel_str)
            mace_check(kernels is not None, "Get kernels failed.")
            size = kernels.ints[0] * kernels.ints[1]
            if size >= 4:
                return "s4"
        return None

    def op_def_desc_type_matched(self, op_def, op_desc):
        data_format_match = op_desc.data_format is None or \
                            op_desc.data_format == \
                            self.get_op_data_format(op_def)
        if not data_format_match:
            return False
        op_data_type = self.get_op_data_type(op_def)
        data_type_match = \
            op_desc.data_type is None or \
            op_desc.data_type == op_data_type or \
            (op_desc.data_type == mace_pb2.DT_FLOAT and
             (op_data_type == mace_pb2.DT_HALF or
              op_data_type == mace_pb2.DT_FLOAT16 or
              op_data_type == mace_pb2.DT_BFLOAT16))
        if not data_type_match:
            return False
        op_tag = self.get_op_tag(op_def)
        if op_tag != op_desc.tag:
            return False
        return True

    def op_def_desc_matched(self, op_def, op_desc):
        if not self.op_def_desc_type_matched(op_def, op_desc):
            return False
        return op_def.name == op_desc.name

    def find_op_in_desc_map(self, op_def, op_desc_map):
        if op_def.type not in op_desc_map:
            return None
        op_descs = op_desc_map[op_def.type]
        for op_desc in op_descs:
            if self.op_def_desc_type_matched(op_def, op_desc):
                return op_desc
        print("The op %s's data type can not be found in op_desc_map" %
              op_def.type)
        return None

    def get_op_desc_map_from_model(self):
        if len(self.op_desc_map) > 0:
            return self.op_desc_map
        op_desc_raw_map = {}
        for i in range(len(McSupportedOps)):
            op_desc = McSupportedOps[i]
            if op_desc.type not in op_desc_raw_map:
                op_desc_raw_map[op_desc.type] = []
            op_desc_raw_map[op_desc.type].append(op_desc)

        self.op_class_name_list = []
        self.op_src_path_list = []
        self.op_desc_map = {}
        idx = 0
        for op_def in self.net_def.op:
            new_op_desc = None
            op_desc = self.find_op_in_desc_map(op_def, self.op_desc_map)
            if op_desc is None:
                new_op_desc = self.find_op_in_desc_map(op_def, op_desc_raw_map)
                mace_check(new_op_desc is not None,
                           "not support op type %s, data type is %s, format is %s" %  # noqa
                           (op_def.type, self.get_op_data_type(op_def),
                            self.get_op_data_format(op_def)))
                if op_def.type not in self.op_desc_map:
                    self.op_desc_map[op_def.type] = []
            else:
                new_op_desc = copy.deepcopy(op_desc)
            new_op_desc.name = op_def.name
            new_op_desc.idx = idx
            idx += 1
            self.op_desc_map[op_def.type].append(new_op_desc)
        return self.op_desc_map

    def get_op_desc_list_from_model(self):
        op_desc_map = self.get_op_desc_map_from_model()
        op_desc_list = []
        for op_descs in op_desc_map.values():
            op_desc_list.extend(op_descs)
        op_desc_list.sort(key=lambda op_desc: op_desc.idx)
        op_class_name_list = [op_desc.class_name for op_desc in op_desc_list]
        op_desc_list.sort(key=lambda op_desc: op_desc.src_path)
        op_src_path_list = [op_desc.src_path for op_desc in op_desc_list]
        return (list(set(op_src_path_list)), op_class_name_list)
