import sys
import operator
from mace.proto import mace_pb2

class MemoryOptimizer(object):
  def __init__(self, net_def):
    self.net_def = net_def
    self.idle_mem = set()
    self.op_mem = {}    # op_name->mem_id
    self.mem_block = {} # mem_id->[x, y]
    self.total_mem_count = 0
    self.ref_counter = {}

    consumers = {}
    for op in net_def.op:
      if self.is_buffer_image_op(op):
        continue
      for ipt in op.input:
        if ipt not in consumers:
          consumers[ipt] = []
        consumers[ipt].append(op)
    # only ref op's output tensor
    for op in net_def.op:
      if self.is_buffer_image_op(op):
        continue
      tensor_name = self._op_to_tensor(op)
      if tensor_name in consumers:
        self.ref_counter[tensor_name] = len(consumers[tensor_name])
      else:
        self.ref_counter[tensor_name] = 0

  def _op_to_tensor(self, op):
    return op.name + ':0'

  def is_buffer_image_op(self, op):
    return op.type == 'BufferToImage' or op.type == 'ImageToBuffer'

  def optimize(self):
    for op in self.net_def.op:
      if self.is_buffer_image_op(op):
        continue
      if len(self.idle_mem) == 0:
        # allocate new mem
        mem_id = self.total_mem_count
        self.total_mem_count += 1
      else:
        # reuse mem
        mem_id = self.idle_mem.pop()

      op.mem_id = mem_id
      self.op_mem[self._op_to_tensor(op)] = mem_id
      if mem_id not in self.mem_block:
        self.mem_block[mem_id] = [0, 0]
      mem_size = self.mem_block[mem_id]
      mem_size[1] = max(mem_size[1], op.output_shape[0].dims[0] * op.output_shape[0].dims[1])
      mem_size[0] = max(mem_size[0], op.output_shape[0].dims[2] * (op.output_shape[0].dims[3]+3)/4)

      # de-ref input tensor mem
      for ipt in op.input:
        if ipt in self.ref_counter:
          self.ref_counter[ipt] -= 1
          if self.ref_counter[ipt] == 0:
            self.idle_mem.add(self.op_mem[ipt])
          elif self.ref_counter[ipt] < 0:
            raise Exception('ref count is less than 0')

    for mem in self.mem_block:
      arena = self.net_def.mem_arena
      block = arena.mem_block.add()
      block.mem_id = mem
      block.x = self.mem_block[mem][0]
      block.y = self.mem_block[mem][1]

    print('total op: %d', len(self.net_def.op))
    origin_mem_size = 0
    optimized_mem_size = 0
    for op in self.net_def.op:
      if self.is_buffer_image_op(op):
        continue
      origin_mem_size += reduce(operator.mul, op.output_shape[0].dims, 1)
    for mem in self.mem_block:
      optimized_mem_size += reduce(operator.mul, self.mem_block[mem], 4)

    print('origin mem: %d, optimized mem: %d', origin_mem_size, optimized_mem_size)


def optimize_memory(net_def):
  mem_optimizer = MemoryOptimizer(net_def)
  mem_optimizer.optimize()