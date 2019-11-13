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


from google.protobuf.descriptor import FieldDescriptor
from utils.util import mace_check
import sys
import struct
import tempfile

if sys.version > '3':
    import queue
else:
    import Queue as queue

SimpleTypeArray = [
    FieldDescriptor.TYPE_DOUBLE,
    FieldDescriptor.TYPE_FLOAT,
    FieldDescriptor.TYPE_INT64,
    FieldDescriptor.TYPE_UINT64,
    FieldDescriptor.TYPE_INT32,
    FieldDescriptor.TYPE_BOOL,
    FieldDescriptor.TYPE_UINT32,
    FieldDescriptor.TYPE_ENUM,
]

# This type is string but it should be stored specially
TYPE_STRING_EX = FieldDescriptor.MAX_TYPE + 1000
TYPE_BYTES_EX = FieldDescriptor.MAX_TYPE + 1001
TYPE_UINT16 = FieldDescriptor.MAX_TYPE + 1002


class ObjInfo:
    def __init__(self, obj, parent_addr, offset, type=None):
        self.obj = obj
        self.parent_addr = parent_addr
        self.offset = offset
        self.type = type


class ProtoConverter:
    def __init__(self, offset16=False, write_magic=False, exclude_fileds={}):
        self.offset16 = offset16
        self.write_magic = write_magic
        self.exclude_fileds = exclude_fileds

    # return the length of string with '\0'
    def str_raw_len(self, str):
        length = len(str)
        if length > 0:
            length += 1
        return length

    # return the string length which can by devided by 4
    def str_pack_len(self, str):
        return int((self.str_raw_len(str) + 3) / 4) * 4

    def pack(self, value, pb_type):
        if pb_type is FieldDescriptor.TYPE_INT32 or \
                pb_type is FieldDescriptor.TYPE_INT64:
            return struct.pack('<i', value)
        elif pb_type is FieldDescriptor.TYPE_UINT32 or \
                pb_type is FieldDescriptor.TYPE_ENUM or \
                pb_type is FieldDescriptor.TYPE_UINT64:
            return struct.pack('<I', value)
        elif pb_type is FieldDescriptor.TYPE_BOOL:
            return struct.pack('<i', (int)(value))
        elif pb_type is FieldDescriptor.TYPE_FLOAT:
            return struct.pack('<f', value)
        elif pb_type is FieldDescriptor.TYPE_DOUBLE:
            return struct.pack('<d', value)
        elif pb_type is TYPE_UINT16:
            return struct.pack('<H', value)
        elif pb_type is FieldDescriptor.TYPE_STRING or \
                pb_type is FieldDescriptor.TYPE_BYTES:
            if isinstance(value, str):
                value = bytes(value.encode('utf-8'))
            length = self.str_raw_len(value)
            if length == 0:
                return b''
            pack_length = self.str_pack_len(value)
            empty_len = pack_length - length
            while empty_len > 0:
                value += b'\0'
                empty_len -= 1
            return struct.pack('<' + str(pack_length) + 's', value)
        else:
            mace_check(False,
                       "The pack's pb_type is not supported: %s" % pb_type)

    def get_pack_type(self):
        pack_type = FieldDescriptor.TYPE_UINT32
        if self.offset16:
            pack_type = TYPE_UINT16
        return pack_type

    def bs_info_to_bytes(self, in_bytes, bs,
                         object_queue, parent_addr, type):
        length = self.str_pack_len(bs)
        in_bytes += self.pack(length, self.get_pack_type())
        offset = len(in_bytes)
        in_bytes += self.pack(offset, self.get_pack_type())
        if length > 0:
            object_queue.put(ObjInfo(bs, parent_addr, offset, type))
        return in_bytes

    def string_info_to_bytes(self, in_bytes, string,
                             object_queue, parent_addr):
        return self.bs_info_to_bytes(in_bytes, string, object_queue,
                                     parent_addr, FieldDescriptor.TYPE_STRING)

    def bytes_info_to_bytes(self, in_bytes, bytes, object_queue, parent_addr):
        return self.bs_info_to_bytes(in_bytes, bytes, object_queue,
                                     parent_addr, FieldDescriptor.TYPE_BYTES)

    def array_to_bytes(self, in_bytes, array,
                       object_queue, parent_addr, descriptor):
        length = len(array)
        in_bytes += self.pack(length, self.get_pack_type())
        offset = len(in_bytes)
        in_bytes += self.pack(offset, self.get_pack_type())
        if length > 0:
            array_length = len(array)
            for i in range(array_length):
                # other units needn't write offset to their parent
                array_parent_addr = parent_addr
                if i > 0:
                    array_parent_addr = -1
                des_type = descriptor.type
                if des_type is FieldDescriptor.TYPE_STRING:
                    des_type = TYPE_STRING_EX
                elif des_type is FieldDescriptor.TYPE_BYTES:
                    des_type = TYPE_BYTES_EX
                object_queue.put(
                    ObjInfo(array[i], array_parent_addr, offset, des_type))
        return in_bytes

    def container_obj_to_bytes(self, obj_info, object_queue, parent_addr):
        bytes = b''
        if self.write_magic:
            bytes = struct.pack('<4s', obj_info.obj.DESCRIPTOR.name[0:4])

        for descriptor in obj_info.obj.DESCRIPTOR.fields:
            if obj_info.obj.DESCRIPTOR.name in self.exclude_fileds and \
                    descriptor.name in self.exclude_fileds[
                obj_info.obj.DESCRIPTOR.name]:  # noqa
                continue
            value = getattr(obj_info.obj, descriptor.name)
            if descriptor.label == descriptor.LABEL_REPEATED:
                array = value
                bytes = self.array_to_bytes(bytes, array, object_queue,
                                            parent_addr, descriptor)
            elif descriptor.type in SimpleTypeArray:
                bytes += self.pack(value, descriptor.type)
            elif descriptor.type is descriptor.TYPE_STRING:
                bytes = self.string_info_to_bytes(bytes, value, object_queue,
                                                  parent_addr)
            elif descriptor.type is descriptor.TYPE_BYTES:
                bytes = self.bytes_info_to_bytes(bytes, value, object_queue,
                                                 parent_addr)
            else:
                mace_check(
                    False,
                    "The pb type is not supported: %s" % descriptor.type)
        return bytes

    def object_to_bytes(self, obj_info, object_queue, start_addr):
        if hasattr(obj_info.obj, 'DESCRIPTOR'):
            obj_bytes = self.container_obj_to_bytes(obj_info, object_queue,
                                                    start_addr)
        elif obj_info.type is FieldDescriptor.TYPE_STRING:
            obj_bytes = self.pack(bytes(obj_info.obj.encode('utf-8')),
                                  obj_info.type)
        elif obj_info.type is FieldDescriptor.TYPE_BYTES:
            obj_bytes = self.pack(obj_info.obj, obj_info.type)
        elif obj_info.type is TYPE_STRING_EX:
            obj_bytes = self.string_info_to_bytes(b'', obj_info.obj,
                                                  object_queue, start_addr)
        elif obj_info.type is TYPE_BYTES_EX:
            obj_bytes = self.bytes_info_to_bytes(b'', obj_info.obj,
                                                 object_queue, start_addr)
        else:  # simple obj
            obj_bytes = self.pack(obj_info.obj, obj_info.type)
        return obj_bytes

    def write_obj_queue_to_file(self, object_queue, f):
        while not object_queue.empty():
            obj_info = object_queue.get()
            start_addr = f.tell()
            bytes = self.object_to_bytes(obj_info, object_queue, start_addr)
            f.write(bytes)

            # write the obj's offset in its parent
            if obj_info.parent_addr >= 0:
                end_addr = f.tell()
                f.seek(obj_info.parent_addr + obj_info.offset, 0)
                f.write(self.pack(start_addr - obj_info.parent_addr,
                                  self.get_pack_type()))
                f.seek(end_addr, 0)

    def proto_to_bytes(self, root_obj, ):
        object_queue = queue.Queue()
        object_queue.put(ObjInfo(root_obj, -1, -1))
        with tempfile.TemporaryFile() as f:
            self.write_obj_queue_to_file(object_queue, f)
            f.seek(0)
            return f.read()
        return None
