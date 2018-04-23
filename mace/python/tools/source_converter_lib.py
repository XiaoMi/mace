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

import os
import uuid
import numpy as np
import hashlib

from mace.proto import mace_pb2
from jinja2 import Environment, FileSystemLoader

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


def rename_tensor(net_def):
    tensor_map = {}
    for t in net_def.tensors:
        if t.name not in tensor_map:
            tensor_map[t.name] = "_" + t.name[:-2].replace("/", "_")
            t.name = tensor_map[t.name]
    for op in net_def.op:
        for i in range(len(op.input)):
            if op.input[i] in tensor_map:
                op.input[i] = tensor_map[op.input[i]]
        for i in range(len(op.output)):
            if op.output[i] in tensor_map:
                op.output[i] = tensor_map[op.output[i]]


class TensorInfo:
    def __init__(self, id, t, runtime):
        self.id = id
        self.data_type = mace_pb2.DataType.Name(t.data_type)
        if t.data_type == mace_pb2.DT_FLOAT:
            if runtime == 'gpu':
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


def stringfy(value):
    return ', '.join('"{0}"'.format(w) for w in value)


def convert_to_source(net_def, mode_pb_checksum, template_dir, obfuscate,
                      model_tag, output, runtime, embed_model_data):
    if obfuscate:
        obfuscate_name(net_def)
    else:
        rename_tensor(net_def)

    # Capture our current directory
    print template_dir

    # Create the jinja2 environment.
    j2_env = Environment(
        loader=FileSystemLoader(template_dir), trim_blocks=True)
    j2_env.filters['stringfy'] = stringfy
    output_dir = os.path.dirname(output) + '/'
    # generate tensor source files
    template_name = 'tensor_source.jinja2'
    model_data = []
    offset = 0
    counter = 0
    for t in net_def.tensors:
        tensor_info = TensorInfo(counter, t, runtime)
        # align
        if tensor_info.data_type != 'DT_UINT8' and offset % 4 != 0:
            padding = 4 - offset % 4
            model_data.extend(bytearray([0] * padding))
            offset += padding
        source = j2_env.get_template(template_name).render(
            tensor_info=tensor_info,
            tensor=t,
            tag=model_tag,
            runtime=runtime,
            offset=offset,
        )
        model_data.extend(tensor_info.data)
        offset += len(tensor_info.data)
        with open(output_dir + 'tensor' + str(counter) + '.cc', "wb") as f:
            f.write(source)
        counter += 1

    # generate tensor data
    template_name = 'tensor_data.jinja2'
    source = j2_env.get_template(template_name).render(
        tag=model_tag,
        embed_model_data=embed_model_data,
        model_data_size=offset,
        model_data=model_data)
    with open(output_dir + 'tensor_data' + '.cc', "wb") as f:
        f.write(source)
    if not embed_model_data:
        with open(output_dir + model_tag + '.data', "wb") as f:
            f.write(bytearray(model_data))

    # generate op source files
    template_name = 'operator.jinja2'
    counter = 0
    op_size = len(net_def.op)
    for start in range(0, op_size, 10):
        source = j2_env.get_template(template_name).render(
            start=start,
            end=min(start + 10, op_size),
            net=net_def,
            tag=model_tag,
            runtime=runtime,
        )
        with open(output_dir + 'op' + str(counter) + '.cc', "wb") as f:
            f.write(source)
        counter += 1

    # generate model source files
    template_name = 'model.jinja2'
    tensors = [
        TensorInfo(i, net_def.tensors[i], runtime)
        for i in range(len(net_def.tensors))
    ]
    source = j2_env.get_template(template_name).render(
        tensors=tensors,
        net=net_def,
        tag=model_tag,
        runtime=runtime,
        model_pb_checksum=mode_pb_checksum)
    with open(output, "wb") as f:
        f.write(source)

    # generate model header file
    template_name = 'model_header.jinja2'
    source = j2_env.get_template(template_name).render(tag=model_tag, )
    with open(output_dir + model_tag + '.h', "wb") as f:
        f.write(source)
