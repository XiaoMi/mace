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

import hashlib
import numpy as np

from mace.proto import mace_pb2

GENERATED_NAME = set()


def generate_obfuscated_name(namespace, name):
    md5 = hashlib.md5()
    md5.update(namespace)
    md5.update(name)
    md5_digest = md5.hexdigest()

    name = md5_digest[:8]
    while name in GENERATED_NAME:
        name = md5_digest
        assert name not in GENERATED_NAME
    GENERATED_NAME.add(name)
    return name


def generate_tensor_map(tensors):
    tensor_map = {}
    for t in tensors:
        if t.name not in tensor_map:
            tensor_map[t.name] = generate_obfuscated_name("tensor", t.name)
    return tensor_map


def generate_in_out_map(ops, tensor_map):
    in_out_map = {}
    for op in ops:
        op.name = generate_obfuscated_name("op", op.name)
        for input_name in op.input:
            if input_name not in in_out_map:
                if input_name in tensor_map:
                    in_out_map[input_name] = tensor_map[input_name]
                else:
                    in_out_map[input_name] = generate_obfuscated_name(
                        "in", input_name)
        for output_name in op.output:
            if output_name not in in_out_map:
                if output_name in tensor_map:
                    in_out_map[output_name] = tensor_map[output_name]
                else:
                    in_out_map[output_name] = generate_obfuscated_name(
                        "out", output_name)
    return in_out_map


def obfuscate_name(net_def):
    input_node = "mace_input_node"
    output_node = "mace_output_node"
    tensor_map = generate_tensor_map(net_def.tensors)
    in_out_map = generate_in_out_map(net_def.op, tensor_map)
    for t in net_def.tensors:
        if input_node not in t.name and output_node not in t.name:
            t.name = tensor_map[t.name]
    for op in net_def.op:
        for i in range(len(op.input)):
            if input_node not in op.input[i]:
                op.input[i] = in_out_map[op.input[i]]
        for i in range(len(op.output)):
            if output_node not in op.output[i]:
                op.output[i] = in_out_map[op.output[i]]


def normalize_op_name(op_name):
    idx = op_name.rfind(':')
    if idx == -1:
        return op_name
    else:
        return op_name[:idx]


def rename_tensor(net_def):
    tensor_map = {}
    for t in net_def.tensors:
        if t.name not in tensor_map:
            tensor_map[t.name] = "_" + normalize_op_name(t.name).replace("/",
                                                                         "_")
            t.name = tensor_map[t.name]
    for op in net_def.op:
        for i in range(len(op.input)):
            if op.input[i] in tensor_map:
                op.input[i] = tensor_map[op.input[i]]
        for i in range(len(op.output)):
            if op.output[i] in tensor_map:
                op.output[i] = tensor_map[op.output[i]]


class TensorInfo:
    def __init__(self, id, t, runtime, gpu_data_type):
        self.id = id
        self.data_type = mace_pb2.DataType.Name(t.data_type)
        if t.data_type == mace_pb2.DT_FLOAT:
            if runtime == 'gpu' and gpu_data_type == 'half':
                self.data_type = mace_pb2.DT_HALF
                self.data = bytearray(
                    np.array(t.float_data).astype(np.float16).tobytes())
            else:
                self.data_type = mace_pb2.DT_FLOAT
                self.data = bytearray(
                    np.array(t.float_data).astype(np.float32).tobytes())
        elif t.data_type == mace_pb2.DT_INT32:
            self.data = bytearray(
                np.array(t.int32_data).astype(np.int32).tobytes())
        elif t.data_type == mace_pb2.DT_UINT8:
            self.data = bytearray(
                np.array(t.int32_data).astype(np.uint8).tolist())
        else:
            raise Exception('Tensor data type %s not supported' % t.data_type)


def get_tensor_info_and_model_data(net_def, runtime, gpu_data_type):
    model_data = []
    offset = 0
    counter = 0
    tensor_infos = []
    for t in net_def.tensors:
        tensor_info = TensorInfo(counter, t, runtime, gpu_data_type)
        tensor_infos.append(tensor_info)
        # align
        if tensor_info.data_type != 'DT_UINT8' and offset % 4 != 0:
            padding = 4 - offset % 4
            model_data.extend(bytearray([0] * padding))
            offset += padding

        if t.data_type == mace_pb2.DT_FLOAT:
            t.data_size = len(t.float_data)
        elif t.data_type == mace_pb2.DT_INT32:
            t.data_size = len(t.int32_data)
        elif t.data_type == mace_pb2.DT_UINT8:
            t.data_size = len(t.int32_data)
        t.offset = offset

        counter += 1
        model_data.extend(tensor_info.data)
        offset += len(tensor_info.data)

    return tensor_infos, model_data


def del_tensor_data(net_def, runtime, gpu_data_type):
    for t in net_def.tensors:
        if t.data_type == mace_pb2.DT_FLOAT:
            del t.float_data[:]
        elif t.data_type == mace_pb2.DT_INT32:
            del t.int32_data[:]
        elif t.data_type == mace_pb2.DT_UINT8:
            del t.int32_data[:]


def update_tensor_data_type(net_def, runtime, gpu_data_type):
    for t in net_def.tensors:
        if t.data_type == mace_pb2.DT_FLOAT and runtime == 'gpu' \
                and gpu_data_type == 'half':
            t.data_type = mace_pb2.DT_HALF
