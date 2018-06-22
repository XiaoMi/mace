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

import sys
import operator
from mace.proto import mace_pb2


class MemoryOptimizer(object):
    def __init__(self, net_def):
        self.net_def = net_def
        self.idle_mem = set()
        self.op_mem = {}  # op_name->mem_id
        self.mem_block = {}  # mem_id->[size] or mem_id->[x, y]
        self.total_mem_count = 0
        self.input_ref_counter = {}
        self.mem_ref_counter = {}

        consumers = {}
        for op in net_def.op:
            if not self.op_need_optimize_memory(op):
                continue
            for ipt in op.input:
                if ipt not in consumers:
                    consumers[ipt] = []
                consumers[ipt].append(op)
        # only ref op's output tensor
        for op in net_def.op:
            if not self.op_need_optimize_memory(op):
                continue
            for output in op.output:
                tensor_name = output
                if tensor_name in consumers:
                    self.input_ref_counter[tensor_name] = \
                        len(consumers[tensor_name])
                else:
                    self.input_ref_counter[tensor_name] = 0

    def op_need_optimize_memory(self, op):
        return True

    def get_op_mem_block(self, op_type, output_shape):
        return [reduce(operator.mul, output_shape, 1)]

    def mem_size(self, memory_block):
        return memory_block[0]

    def sub_mem_block(self, mem_block1, mem_block2):
        return self.mem_size(mem_block1) - self.mem_size(mem_block2)

    def resize_mem_block(self, old_mem_block, op_mem_block):
        return [max(old_mem_block[0], op_mem_block[0])]

    def add_net_mem_blocks(self):
        for mem in self.mem_block:
            arena = self.net_def.mem_arena
            block = arena.mem_block.add()
            block.mem_id = mem
            block.x = self.mem_block[mem][0]
            block.y = 1

    def get_total_origin_mem_size(self):
        origin_mem_size = 0
        for op in self.net_def.op:
            if not self.op_need_optimize_memory(op):
                continue
            origin_mem_size += reduce(operator.mul, op.output_shape[0].dims, 1)
        return origin_mem_size

    def get_total_optimized_mem_size(self):
        optimized_mem_size = 0
        for mem in self.mem_block:
            print mem, self.mem_block[mem]
            optimized_mem_size += self.mem_size(self.mem_block[mem])
        return optimized_mem_size

    @staticmethod
    def is_memory_reuse_op(op):
        return op.type == 'Reshape' or op.type == 'Identity' \
               or op.type == 'Squeeze'

    def optimize(self):
        for op in self.net_def.op:
            if not self.op_need_optimize_memory(op):
                continue
            if not op.output_shape:
                print("WARNING: There is no output shape information to "
                      "do memory optimization. %s (%s)" % (op.name, op.type))
                return
            if len(op.output_shape) != len(op.output):
                print('WARNING: the number of output shape is not equal to '
                      'the number of output.')
                return
            for i in range(len(op.output)):
                if self.is_memory_reuse_op(op):
                    # make these ops reuse memory of input tensor
                    mem_id = self.op_mem.get(op.input[0], -1)
                else:
                    op_mem_block = self.get_op_mem_block(
                        op.type,
                        op.output_shape[i].dims)
                    mem_id = -1
                    if len(self.idle_mem) > 0:
                        best_mem_add_size = sys.maxint
                        best_mem_waste_size = sys.maxint
                        for mid in self.idle_mem:
                            old_mem_block = self.mem_block[mid]
                            new_mem_block = self.resize_mem_block(
                                old_mem_block, op_mem_block)
                            add_mem_size = self.sub_mem_block(new_mem_block,
                                                              old_mem_block)
                            waste_mem_size = self.sub_mem_block(new_mem_block,
                                                                op_mem_block)

                            # minimize add_mem_size; if best_mem_add_size is 0,
                            # then minimize waste_mem_size
                            if (best_mem_add_size > 0 and
                                    add_mem_size < best_mem_add_size) \
                                    or (best_mem_add_size == 0 and
                                        waste_mem_size < best_mem_waste_size):
                                best_mem_id = mid
                                best_mem_add_size = add_mem_size
                                best_mem_waste_size = waste_mem_size
                                best_mem_block = new_mem_block

                        # if add mem size < op mem size, then reuse it
                        if best_mem_add_size <= self.mem_size(op_mem_block):
                            self.mem_block[best_mem_id] = best_mem_block
                            mem_id = best_mem_id
                            self.idle_mem.remove(mem_id)

                    if mem_id == -1:
                        mem_id = self.mem_id_base() + self.total_mem_count
                        self.total_mem_count += 1
                        self.mem_block[mem_id] = op_mem_block

                if mem_id != -1:
                    op.mem_id.extend([mem_id])
                    self.op_mem[op.output[i]] = mem_id
                    if mem_id not in self.mem_ref_counter:
                        self.mem_ref_counter[mem_id] = 1
                    else:
                        self.mem_ref_counter[mem_id] += 1

            # de-ref input tensor mem
            for idx in xrange(len(op.input)):
                ipt = op.input[idx]
                if ipt in self.input_ref_counter:
                    self.input_ref_counter[ipt] -= 1
                    if self.input_ref_counter[ipt] == 0 \
                            and ipt in self.op_mem:
                        mem_id = self.op_mem[ipt]
                        self.mem_ref_counter[mem_id] -= 1
                        if self.mem_ref_counter[mem_id] == 0:
                            self.idle_mem.add(self.op_mem[ipt])
                    elif self.input_ref_counter[ipt] < 0:
                        raise Exception('ref count is less than 0')

        self.add_net_mem_blocks()

        print("total op: %d" % len(self.net_def.op))
        print("origin mem: %d, optimized mem: %d" % (
              self.get_total_origin_mem_size(),
              self.get_total_optimized_mem_size()))

    def mem_id_base(self):
        return 0


class GPUMemoryOptimizer(MemoryOptimizer):
    def op_need_optimize_memory(self, op):
        if op.type == 'BufferToImage':
            for arg in op.arg:
                if arg.name == 'mode' and arg.i == 0:
                    return False
        return op.type != 'ImageToBuffer'

    def get_op_mem_block(self, op_type, output_shape):
        mem_block = [0, 0]
        if op_type == 'WinogradTransform' or op_type == 'MatMul':
            mem_block[0] = output_shape[2]
            mem_block[1] = output_shape[0] * int((output_shape[1] + 3) / 4)
        else:
            if len(output_shape) == 2:  # only support fc/softmax
                mem_block[0] = int((output_shape[1] + 3) / 4)
                mem_block[1] = output_shape[0]
            else:
                mem_block[0] = output_shape[2] * int((output_shape[3] + 3) / 4)
                mem_block[1] = output_shape[0] * output_shape[1]
        return mem_block

    def mem_size(self, memory_block):
        return memory_block[0] * memory_block[1] * 4

    def resize_mem_block(self, old_mem_block, op_mem_block):
        resize_mem_block = [
            max(old_mem_block[0], op_mem_block[0]),
            max(old_mem_block[1], op_mem_block[1])
        ]
        return resize_mem_block

    def add_net_mem_blocks(self):
        for mem in self.mem_block:
            arena = self.net_def.mem_arena
            block = arena.mem_block.add()
            block.mem_id = mem
            block.x = self.mem_block[mem][0]
            block.y = self.mem_block[mem][1]

    def mem_id_base(self):
        return 20000


def optimize_gpu_memory(net_def):
    mem_optimizer = GPUMemoryOptimizer(net_def)
    mem_optimizer.optimize()


def optimize_cpu_memory(net_def):
    mem_optimizer = MemoryOptimizer(net_def)
    mem_optimizer.optimize()
