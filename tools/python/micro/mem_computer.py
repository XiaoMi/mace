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

from utils.convert_util import data_type_to_np_dt
from utils.util import mace_check

import numpy as np


class MemBlock:
    def __init__(self, tensor_name, offset, size):
        self.tensor_name = tensor_name
        self.offset = offset
        self.size = size


class MemComputer:
    def __init__(self, net_def, np_data_type):
        self.net_def = net_def
        self.np_data_type = np_data_type
        self.const_tensor_names = []
        for const_tensor in net_def.tensors:
            self.const_tensor_names.append(const_tensor.name)
        self.input_names = []
        for input_info in net_def.input_info:
            self.input_names.append(input_info.name)

    def init_computer(self):
        self.free_mem_list = []
        self.used_mem_list = []
        self.buffer_size = 0
        self.ref_counts = {}
        for op in self.net_def.op:
            for tensor_name in op.input:
                if tensor_name in self.const_tensor_names or \
                        tensor_name in self.input_names:
                    continue
                if tensor_name not in self.ref_counts:
                    self.ref_counts[tensor_name] = 0
                self.ref_counts[tensor_name] += 1

    def get_mem_size(self, op, output_shape):
        np_data_type = self.np_data_type
        if len(op.output_type) > 0:
            np_data_type = \
                data_type_to_np_dt(op.output_type[0], self.np_data_type)
        data_type_bytes = np.dtype(np_data_type).itemsize
        if op.type == 'WinogradTransform' or op.type == 'GEMM':
            mace_check(len(output_shape) == 4,
                       "WinogradTransform and GEMM only support 4-dim")
            mem_size = output_shape[2] * output_shape[3] * output_shape[0] \
                * int((output_shape[1] + 3) / 4) * 4
        else:
            dim_size = len(output_shape)
            if dim_size > 0:
                mem_size = int((output_shape[dim_size - 1] + 3) / 4) * 4
                for i in range(dim_size - 1):
                    mem_size *= output_shape[i]
            else:
                print("the op %s's output dim size is 0" % op.type)
                mem_size = 0
        return mem_size * data_type_bytes

    def remove_mem_block_by_name(self, mem_list, tensor_name):
        return_mem_block = None
        for mem_block in mem_list:
            if tensor_name == mem_block.tensor_name:
                return_mem_block = mem_block
                mem_list.remove(mem_block)
                break
        return return_mem_block

    def fake_new(self, op):
        output_size = len(op.output)
        for i in range(output_size):
            mem_size = self.get_mem_size(op, op.output_shape[i].dims)
            final_mem_block = None
            reused = False
            for mem_block in self.free_mem_list:
                if mem_block.size >= mem_size:
                    mem_block.tensor_name = op.output[i]
                    final_mem_block = mem_block
                    self.free_mem_list.remove(mem_block)
                    mace_check(final_mem_block is not None,
                               "Error: final_mem_block should not be None")
                    reused = True
                    # print("reuse a tensor mem: %s -> %s" %
                    #       (mem_size, mem_block.size))
                    break
            if not reused:
                final_mem_block = MemBlock(op.output[i], self.buffer_size,
                                           mem_size)
                self.buffer_size += mem_size
                # print("new a tensor mem: %s" % final_mem_block.size)

            # for micro, mem_id is mem_offset
            op.mem_id.append(final_mem_block.offset)
            self.used_mem_list.append(final_mem_block)

    def fake_delete(self, op):
        for tensor_name in op.input:
            if tensor_name in self.const_tensor_names or \
                    tensor_name in self.input_names:
                continue
            mace_check(tensor_name in self.ref_counts and
                       self.ref_counts[tensor_name] > 0,
                       "Invalid: ref_count is 0.")
            self.ref_counts[tensor_name] -= 1
            if self.ref_counts[tensor_name] is 0:
                mem_block = self.remove_mem_block_by_name(
                    self.used_mem_list, tensor_name)
                mace_check(mem_block is not None,
                           "error, can not find tensor: %s" % tensor_name)
                self.free_mem_list.append(mem_block)
                self.free_mem_list.sort(key=lambda mem_block: mem_block.size)

    def fake_execute_op(self, op):
        for i in range(len(op.output)):
            self.fake_new(op)
            self.fake_delete(op)

    # return the tensor memory size needed by mace micro
    def compute(self):
        self.init_computer()
        for op in self.net_def.op:
            self.fake_execute_op(op)
        return self.buffer_size
