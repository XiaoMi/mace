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

# python tools/python/convert.py \
# --config ../mace-models/mobilenet-v2/mobilenet-v2.yml

import array
import numpy as np
import struct
from py_proto import mace_pb2


def Float2BFloat16Bytes(float_data):
    int_datas = []
    for value in float_data:
        bytes = struct.pack("f", value)
        int_data = struct.unpack('i', bytes)[0]
        int_datas.append(int_data >> 16)
    return np.array(int_datas).astype(np.uint16).tobytes()


def merge_params(net_def, data_type):
    def tensor_to_bytes(tensor):
        if tensor.data_type == mace_pb2.DT_HALF:
            data = bytearray(
                np.array(tensor.float_data).astype(np.float16).tobytes())
            tensor.data_size = len(tensor.float_data)
        elif tensor.data_type == mace_pb2.DT_FLOAT:
            data = bytearray(
                np.array(tensor.float_data).astype(np.float32).tobytes())
            tensor.data_size = len(tensor.float_data)
        elif tensor.data_type == mace_pb2.DT_INT32:
            data = bytearray(
                np.array(tensor.int32_data).astype(np.int32).tobytes())
            tensor.data_size = len(tensor.int32_data)
        elif tensor.data_type == mace_pb2.DT_UINT8:
            data = bytearray(
                np.array(tensor.int32_data).astype(np.uint8).tolist())
            tensor.data_size = len(tensor.int32_data)
        elif tensor.data_type == mace_pb2.DT_FLOAT16:
            data = bytearray(
                np.array(tensor.float_data).astype(np.float16).tobytes())
            tensor.data_size = len(tensor.float_data)
        elif tensor.data_type == mace_pb2.DT_BFLOAT16:
            data = Float2BFloat16Bytes(tensor.float_data)
            tensor.data_size = len(tensor.float_data)
        else:
            raise Exception('Tensor data type %s not supported' %
                            tensor.data_type)
        return data

    model_data = []
    offset = 0
    for tensor in net_def.tensors:
        if tensor.data_type == mace_pb2.DT_FLOAT:
            tensor.data_type = data_type
        raw_data = tensor_to_bytes(tensor)
        if tensor.data_type != mace_pb2.DT_UINT8 and offset % 4 != 0:
            padding = 4 - offset % 4
            model_data.extend(bytearray([0] * padding))
            offset += padding

        tensor.offset = offset
        model_data.extend(raw_data)
        offset += len(raw_data)

    for tensor in net_def.tensors:
        if tensor.data_type == mace_pb2.DT_FLOAT \
                or tensor.data_type == mace_pb2.DT_HALF \
                or tensor.data_type == mace_pb2.DT_FLOAT16 \
                or tensor.data_type == mace_pb2.DT_BFLOAT16:
            del tensor.float_data[:]
        elif tensor.data_type == mace_pb2.DT_INT32:
            del tensor.int32_data[:]
        elif tensor.data_type == mace_pb2.DT_UINT8:
            del tensor.int32_data[:]

    return net_def, model_data


def data_type_to_np_dt(data_type, default_np_dt):
    if data_type is None:
        return default_np_dt
    elif data_type == mace_pb2.DT_HALF or data_type == mace_pb2.DT_FLOAT16:
        return np.float16
    elif data_type == mace_pb2.DT_INT32:
        return np.int
    elif data_type == mace_pb2.DT_UINT8:
        return np.uint8
    elif data_type == mace_pb2.DT_BFLOAT16:
        return np.uint16
    else:
        return np.float32
