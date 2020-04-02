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

from py_proto import micro_mem_pb2
from utils.util import mace_check


class GraphBuilder:
    def __init__(self, pb_model, op_resolver):
        self.net_def = pb_model
        self.ops_desc_map = op_resolver.get_op_desc_map_from_model()
        self.op_resolver = op_resolver

        self.init_output_cache()
        self.init_const_tensor_cache()
        self.init_model_input_cache()

    def get_op_idx(self, op_def):
        if op_def.type not in self.ops_desc_map:
            return -1
        op_desc_list = self.ops_desc_map[op_def.type]
        for op_desc in op_desc_list:
            if self.op_resolver.op_def_desc_matched(op_def, op_desc):
                return op_desc.idx
        return -1

    def init_output_cache(self):
        model_outputs = []
        for output_info in self.net_def.output_info:
            model_outputs.append(output_info.name)
        self.output_cache = {}
        self.output_infos = []
        for i in range(len(self.net_def.op)):
            op_def = self.net_def.op[i]
            for k in range(len(op_def.output)):
                tensor_name = op_def.output[k]
                output_info_uint = ((i & 0x0000ffff) << 16) | (k & 0x0000ffff)
                if tensor_name in model_outputs:
                    self.output_infos.append(output_info_uint)
                else:
                    self.output_cache[tensor_name] = output_info_uint

    def init_const_tensor_cache(self):
        self.const_tensor_cache = {}
        for i in range(len(self.net_def.tensors)):
            const_tensor = self.net_def.tensors[i]
            self.const_tensor_cache[const_tensor.name] = \
                (0xffff0000 | (i & 0x0000ffff))

    def init_model_input_cache(self):
        self.model_input_cache = {}
        for i in range(len(self.net_def.input_info)):
            input_info = self.net_def.input_info[i]
            self.model_input_cache[input_info.name] = \
                (0xfffe0000 | (i & 0x0000ffff))

    def build(self):
        graph = micro_mem_pb2.Graph()
        graph.output_infos.extend(self.output_infos)
        for op_def in self.net_def.op:
            op_context = graph.op_contexts.add()
            idx = self.get_op_idx(op_def)
            mace_check(idx >= 0, "Error from the OpResolver.")
            op_context.op_idx = idx

            op_with_model_input = False
            for input in op_def.input:
                input_info = 0
                if input in self.output_cache:
                    input_info = self.output_cache[input]
                elif input in self.const_tensor_cache:
                    input_info = self.const_tensor_cache[input]
                elif input in self.model_input_cache:
                    input_info = self.model_input_cache[input]
                    op_with_model_input = True
                else:
                    mace_check(False,
                               "Model error: can not find input(%s)" % input)
                op_context.input_infos.append(input_info)
            if op_with_model_input:
                graph.input_op_idxs.append(idx)

            for output_shape in op_def.output_shape:
                resize_shape = op_context.output_resize_shapes.add()
                for dim in output_shape.dims:
                    resize_shape.dims.append(dim)
        return graph
