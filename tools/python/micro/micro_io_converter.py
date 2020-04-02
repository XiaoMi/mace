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
from transform.base_converter import MaceOp
from utils.util import mace_check
import copy


class MicroIoConverter:
    @staticmethod
    def add_dt_cast_for_bf16(net_def):
        bf16_net_def = copy.deepcopy(net_def)
        op_num = len(bf16_net_def.op)
        for i in range(op_num):
            bf16_net_def.op.pop()
        model_input = {}
        for input_info in net_def.input_info:
            model_input[input_info.name] = input_info.dims
        model_output = {}
        for output_info in net_def.output_info:
            model_output[output_info.name] = output_info.dims
        for op_def in net_def.op:
            op_added = False
            if len(model_input) > 0:
                for i in range(len(op_def.input)):
                    input_name = op_def.input[i]
                    if input_name in model_input:
                        if op_added:
                            next_op = bf16_net_def.op.pop()
                        else:
                            next_op = copy.deepcopy(op_def)
                            op_added = True

                        op_cast = bf16_net_def.op.add()
                        op_cast.name = MaceOp.Cast.name + "_op_" + input_name
                        op_cast.type = MaceOp.Cast.name
                        op_cast.input.append(input_name)
                        trans_output_name = \
                            MaceOp.Cast.name + "_out_" + input_name
                        op_cast.output.append(trans_output_name)
                        data_type_arg = op_cast.arg.add()
                        data_type_arg.name = 'T'
                        data_type_arg.i = mace_pb2.DT_FLOAT
                        op_cast.output_type.append(mace_pb2.DT_BFLOAT16)
                        output_shape = op_cast.output_shape.add()
                        output_shape.dims.extend(model_input[input_name])

                        next_op.input[i] = trans_output_name
                        bf16_net_def.op.append(next_op)
                        model_input.pop(input_name)
            if len(model_output) > 0:
                mace_check(len(op_def.output) == 1,
                           "Not support output num > 1")
                output_name = op_def.output[0]
                if output_name in model_output:
                    if not op_added:
                        last_op = copy.deepcopy(op_def)
                        op_added = True
                    else:
                        last_op = bf16_net_def.op.pop()
                    last_op.output[0] = output_name + "_" + MaceOp.Cast.name
                    bf16_net_def.op.append(last_op)

                    op_cast = bf16_net_def.op.add()
                    op_cast.name = MaceOp.Cast.name + "_op_" + output_name
                    op_cast.type = MaceOp.Cast.name
                    op_cast.input.append(last_op.output[0])
                    op_cast.output.append(output_name)
                    data_type_arg = op_cast.arg.add()
                    data_type_arg.name = 'T'
                    data_type_arg.i = mace_pb2.DT_BFLOAT16
                    op_cast.output_type.append(mace_pb2.DT_FLOAT)
                    output_shape = op_cast.output_shape.add()
                    output_shape.dims.extend(model_output[output_name])
                    model_output.pop(output_name)
            if not op_added:
                bf16_net_def.op.append(copy.deepcopy(op_def))
        return bf16_net_def

    @staticmethod
    def convert(net_def, data_type):
        if data_type == mace_pb2.DT_BFLOAT16:
            print("data type is bfloat16, add input/output layers")
            return MicroIoConverter.add_dt_cast_for_bf16(net_def)
        else:
            print("data type is %s" % data_type)
        return net_def
