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

from py_proto import mace_pb2
from utils.config_parser import ModelKeys
from utils.util import mace_check
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp


class ScratchComputer:
    def __init__(self, net_def, model_conf):
        self.net_def = net_def
        if ModelKeys.quantize in model_conf and \
                model_conf[ModelKeys.quantize] == 1:
            self.default_data_type = mace_pb2.DT_UINT8
        else:
            self.default_data_type = mace_pb2.DT_FLOAT
        self._scratch_map = {
            MaceOp.Conv2D: self.scratch_size_no_need,
            MaceOp.Squeeze: self.scratch_size_of_squeeze,
            MaceOp.Softmax: self.scratch_size_no_need,
            MaceOp.Eltwise: self.scratch_size_no_need,
            MaceOp.Activation: self.scratch_size_no_need,
            MaceOp.StridedSlice: self.scratch_size_no_need,
            MaceOp.Reduce: self.scratch_size_no_need,
            MaceOp.Stack: self.scratch_size_no_need,
            MaceOp.BiasAdd: self.scratch_size_no_need,
            MaceOp.BatchNorm: self.scratch_size_no_need,
            MaceOp.Shape: self.scratch_size_no_need,
            MaceOp.Reshape: self.scratch_size_no_need,
            MaceOp.ExpandDims: self.scratch_size_of_expand_dims,
            MaceOp.MatMul: self.scratch_size_of_matmul,
            MaceOp.Pooling: self.scratch_size_of_pooling,
            MaceOp.DepthwiseConv2d: self.scratch_size_of_depthwise_conv,
            MaceOp.ArgMax: self.scratch_size_no_need,
            MaceOp.Cast: self.scratch_size_no_need,
        }

    def compute_size(self):
        scratch_size = 1
        for op_def in self.net_def.op:
            mace_check(op_def.type in self._scratch_map,
                       "The %s's scratch func is lost." % op_def.type)
            size = self._scratch_map[op_def.type](op_def)
            if scratch_size < size:
                scratch_size = size
        print("micro scatch buffer size is: %s" % scratch_size)
        return scratch_size

    def scratch_size_no_need(self, op_def):
        return 0

    def get_op_data_type(self, op_def):
        arg = self.get_op_def_arg(op_def, MaceKeyword.mace_op_data_type_str)
        if arg is None:
            return self.default_data_type
        else:
            return arg.i

    def get_data_bytes(self, data_type):
        if data_type == mace_pb2.DT_FLOAT or \
                data_type == mace_pb2.DT_INT32:
            return 4
        elif data_type == mace_pb2.DT_HALF or \
                data_type == mace_pb2.DT_BFLOAT16 or \
                data_type == mace_pb2.DT_FLOAT16:
            return 2
        elif data_type == mace_pb2.DT_UINT8:
            return 1
        else:
            mace_check(False, "Invalid data type: %s" % data_type)

    def scratch_size_of_expand_dims(self, op_def):
        output_dim_size = len(op_def.output_shape[0].dims)
        data_type_bytes = self.get_data_bytes(mace_pb2.DT_INT32)
        return output_dim_size * data_type_bytes

    def scratch_size_of_matmul(self, op_def):
        output_dim_size = len(op_def.output_shape[0].dims)
        data_type_bytes = self.get_data_bytes(mace_pb2.DT_INT32)
        return output_dim_size * data_type_bytes

    def get_op_input_dims(self, op_def, idx):
        input_name = op_def.input[idx]
        for const_tensor in self.net_def.tensors:
            if input_name == const_tensor.name:
                return const_tensor.dims
        for pre_op in self.net_def.op:
            for i in range(len(pre_op.output)):
                if pre_op.output[i] == input_name:
                    return pre_op.output_shape[i].dims
        return None

    def scratch_size_of_pooling(self, op_def):
        input0_dims = self.get_op_input_dims(op_def, 0)
        channels = input0_dims[3]
        mace_check(channels > 0,
                   "can not inference pooling's input shape.")

        int_bytes = self.get_data_bytes(mace_pb2.DT_INT32)
        float_bytes = self.get_data_bytes(mace_pb2.DT_FLOAT)

        return channels * (int_bytes + float_bytes)

    def scratch_size_of_depthwise_conv(self, op_def):
        filter_dims = self.get_op_input_dims(op_def, 1)
        k_batch = filter_dims[0]
        block_size = k_batch
        if block_size > 4:
            block_size = 4
        k_channels = filter_dims[3]
        float_bytes = self.get_data_bytes(mace_pb2.DT_FLOAT)
        return block_size * 4 * k_channels * float_bytes

    def scratch_size_of_squeeze(self, op_def):
        input0_dims = self.get_op_input_dims(op_def, 0)
        return len(input0_dims) * self.get_data_bytes(mace_pb2.DT_FLOAT)
