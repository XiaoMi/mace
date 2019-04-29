# Copyright 2018 The MACE Authors. All Rights Reserved.
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

import datetime
import os
import six
import uuid
import numpy as np
import hashlib
from enum import Enum

from mace.proto import mace_pb2
from mace.python.tools.converter_tool import base_converter as cvt
from mace.python.tools.convert_util import mace_check
from jinja2 import Environment, FileSystemLoader

GENERATED_NAME = set()


class ModelFormat(object):
    file = "file"
    code = "code"


def generate_obfuscated_name(namespace, name):
    md5 = hashlib.md5()
    md5.update(six.b(namespace))
    md5.update(six.b(name))
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


def obfuscate_name(option, net_def):
    input_nodes = set()
    for name in option.input_nodes:
        input_nodes.add(name)
    output_nodes = set()
    for name in option.output_nodes:
        output_nodes.add(name)
    tensor_map = generate_tensor_map(net_def.tensors)
    in_out_map = generate_in_out_map(net_def.op, tensor_map)
    for t in net_def.tensors:
        if t.name not in input_nodes and t.name not in output_nodes:
            t.name = tensor_map[t.name]
    for op in net_def.op:
        for i in range(len(op.input)):
            if op.input[i] not in input_nodes:
                op.input[i] = in_out_map[op.input[i]]
        for i in range(len(op.output)):
            if op.output[i] not in output_nodes:
                op.output[i] = in_out_map[op.output[i]]


def stringfy(value):
    return ', '.join('"{0}"'.format(w) for w in value)


class TensorInfo:
    def __init__(self, id, tensor):
        self.id = id
        self.data_type = tensor.data_type
        if tensor.data_type == mace_pb2.DT_HALF:
            self.data = bytearray(
                np.array(tensor.float_data).astype(np.float16).tobytes())
        elif tensor.data_type == mace_pb2.DT_FLOAT:
            self.data = bytearray(
                np.array(tensor.float_data).astype(np.float32).tobytes())
        elif tensor.data_type == mace_pb2.DT_INT32:
            self.data = bytearray(
                np.array(tensor.int32_data).astype(np.int32).tobytes())
        elif tensor.data_type == mace_pb2.DT_UINT8:
            self.data = bytearray(
                np.array(tensor.int32_data).astype(np.uint8).tolist())
        elif tensor.data_type == mace_pb2.DT_FLOAT16:
            self.data = bytearray(
                np.array(tensor.float_data).astype(np.float16).tobytes())
        else:
            raise Exception('Tensor data type %s not supported' %
                            tensor.data_type)


def update_tensor_infos(net_def, data_type):
    offset = 0
    counter = 0
    tensor_infos = []
    for tensor in net_def.tensors:
        if tensor.data_type == mace_pb2.DT_FLOAT:
            tensor.data_type = data_type

        # Add offset and data_size
        tensor_info = TensorInfo(counter, tensor)
        tensor_infos.append(tensor_info)
        # align
        if tensor_info.data_type != mace_pb2.DT_UINT8 and offset % 4 != 0:
            padding = 4 - offset % 4
            offset += padding

        if tensor.data_type == mace_pb2.DT_FLOAT \
                or tensor.data_type == mace_pb2.DT_HALF \
                or tensor.data_type == mace_pb2.DT_FLOAT16:
            tensor.data_size = len(tensor.float_data)
        elif tensor.data_type == mace_pb2.DT_INT32:
            tensor.data_size = len(tensor.int32_data)
        elif tensor.data_type == mace_pb2.DT_UINT8:
            tensor.data_size = len(tensor.int32_data)
        tensor.offset = offset
        offset += len(tensor_info.data)
        counter += 1


def extract_model_data(net_def):
    model_data = []
    offset = 0
    counter = 0
    for tensor in net_def.tensors:
        tensor_info = TensorInfo(counter, tensor)
        # align
        mace_check(offset <= tensor.offset,
                   "Current offset should be <= tensor.offset")
        if offset < tensor.offset:
            model_data.extend(bytearray([0] * (tensor.offset - offset)))
            offset = tensor.offset
        model_data.extend(tensor_info.data)
        offset += len(tensor_info.data)
        counter += 1
    return model_data


def save_model_data(net_def, model_tag, output_dir):
    model_data = extract_model_data(net_def)
    # generate tensor data
    with open(output_dir + model_tag + '.data', "wb") as f:
        f.write(bytearray(model_data))


def save_model_to_proto(net_def, model_tag, output_dir):
    for tensor in net_def.tensors:
        if tensor.data_type == mace_pb2.DT_FLOAT \
                or tensor.data_type == mace_pb2.DT_HALF \
                or tensor.data_type == mace_pb2.DT_FLOAT16:
            del tensor.float_data[:]
        elif tensor.data_type == mace_pb2.DT_INT32:
            del tensor.int32_data[:]
        elif tensor.data_type == mace_pb2.DT_UINT8:
            del tensor.int32_data[:]
    proto_file_path = output_dir + model_tag + '.pb'
    with open(proto_file_path, "wb") as f:
        f.write(net_def.SerializeToString())
    with open(proto_file_path + '_txt', "w") as f:
        f.write(str(net_def))

    return proto_file_path


def save_model_to_code(net_def, model_tag, device,
                       template_dir, output_dir, embed_model_data,
                       model_checksum, weight_checksum,
                       obfuscate, winograd_conv):
    # Create the jinja2 environment.
    j2_env = Environment(
        loader=FileSystemLoader(template_dir), trim_blocks=True)
    j2_env.filters['stringfy'] = stringfy

    # generate tensor source files
    template_name = 'tensor_source.jinja2'

    counter = 0
    for tensor in net_def.tensors:
        tensor_info = TensorInfo(counter, tensor)
        # convert tensor
        source = j2_env.get_template(template_name).render(
            tensor_info=tensor_info,
            tensor=tensor,
            tag=model_tag,
        )
        with open(output_dir + 'tensor' + str(counter) + '.cc', "w") as f:
            f.write(source)
        counter += 1

    # generate tensor data
    if embed_model_data:
        model_data = extract_model_data(net_def)
        template_name = 'tensor_data.jinja2'
        source = j2_env.get_template(template_name).render(
            tag=model_tag,
            model_data_size=len(model_data),
            model_data=model_data)
        with open(output_dir + 'tensor_data' + '.cc', "w") as f:
            f.write(source)

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
            device=device,
        )
        with open(output_dir + 'op' + str(counter) + '.cc', "w") as f:
            f.write(source)
        counter += 1

    # generate model source files
    build_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    template_name = 'model.jinja2'
    checksum = model_checksum
    if weight_checksum is not None:
        checksum = "{},{}".format(model_checksum, weight_checksum)
    source = j2_env.get_template(template_name).render(
        net=net_def,
        tag=model_tag,
        obfuscate=obfuscate,
        embed_model_data=embed_model_data,
        winograd_conv=winograd_conv,
        checksum=checksum,
        build_time=build_time)
    with open(output_dir + 'model.cc', "w") as f:
        f.write(source)

    # generate model header file
    template_name = 'model_header.jinja2'
    source = j2_env.get_template(template_name).render(tag=model_tag, )
    with open(output_dir + model_tag + '.h', "w") as f:
        f.write(source)


def save_model(option, net_def, model_checksum, weight_checksum, template_dir,
               obfuscate, model_tag, output_dir, embed_model_data,
               winograd_conv, model_graph_format):
    if obfuscate:
        obfuscate_name(option, net_def)

    output_dir = output_dir + '/'
    # update tensor type
    update_tensor_infos(net_def, option.data_type)

    if model_graph_format == ModelFormat.file or not embed_model_data:
        save_model_data(net_def, model_tag, output_dir)

    if model_graph_format == ModelFormat.file:
        save_model_to_proto(net_def, model_tag, output_dir)
    else:
        save_model_to_code(net_def, model_tag, option.device,
                           template_dir, output_dir, embed_model_data,
                           model_checksum, weight_checksum,
                           obfuscate, winograd_conv)
