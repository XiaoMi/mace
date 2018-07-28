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

import enum


def mace_check(condition, msg):
    if not condition:
        raise Exception(msg)


def roundup_div4(value):
    return int((value + 3) / 4)


class OpenCLBufferType(enum.Enum):
    CONV2D_FILTER = 0
    IN_OUT_CHANNEL = 1
    ARGUMENT = 2
    IN_OUT_HEIGHT = 3
    IN_OUT_WIDTH = 4
    WINOGRAD_FILTER = 5
    DW_CONV2D_FILTER = 6
    WEIGHT_HEIGHT = 7
    WEIGHT_WIDTH = 8


def calculate_image_shape(buffer_type, shape, winograd_blk_size=0):
    # keep the same with mace/kernel/opencl/helper.cc
    image_shape = [0, 0]
    if buffer_type == OpenCLBufferType.CONV2D_FILTER:
        mace_check(len(shape) == 4, "Conv2D filter buffer should be 4D")
        image_shape[0] = shape[1]
        image_shape[1] = shape[2] * shape[3] * roundup_div4(shape[0])
    elif buffer_type == OpenCLBufferType.IN_OUT_CHANNEL:
        mace_check(len(shape) == 4, "Conv2D input/output buffer should be 4D")
        image_shape[0] = roundup_div4(shape[3]) * shape[2]
        image_shape[1] = shape[0] * shape[1]
    elif buffer_type == OpenCLBufferType.ARGUMENT:
        mace_check(len(shape) == 1,
                   "Argument buffer should be 1D not " + str(shape))
        image_shape[0] = roundup_div4(shape[0])
        image_shape[1] = 1
    elif buffer_type == OpenCLBufferType.IN_OUT_HEIGHT:
        mace_check(len(shape) == 4, "Input/output buffer should be 4D")
        image_shape[0] = shape[2] * shape[3]
        image_shape[1] = shape[0] * roundup_div4(shape[1])
    elif buffer_type == OpenCLBufferType.IN_OUT_WIDTH:
        mace_check(len(shape) == 4, "Input/output buffer should be 4D")
        image_shape[0] = roundup_div4(shape[2]) * shape[3]
        image_shape[1] = shape[0] * shape[1]
    elif buffer_type == OpenCLBufferType.WINOGRAD_FILTER:
        mace_check(len(shape) == 4, "Winograd filter buffer should be 4D")
        image_shape[0] = roundup_div4(shape[1])
        image_shape[1] = (shape[0] * (winograd_blk_size + 2)
                          * (winograd_blk_size + 2))
    elif buffer_type == OpenCLBufferType.DW_CONV2D_FILTER:
        mace_check(len(shape) == 4, "Winograd filter buffer should be 4D")
        image_shape[0] = shape[0] * shape[2] * shape[3]
        image_shape[1] = roundup_div4(shape[1])
    elif buffer_type == OpenCLBufferType.WEIGHT_HEIGHT:
        mace_check(len(shape) == 4, "Weight buffer should be 4D")
        image_shape[0] = shape[1] * shape[2] * shape[3]
        image_shape[1] = roundup_div4(shape[0])
    elif buffer_type == OpenCLBufferType.WEIGHT_WIDTH:
        mace_check(len(shape) == 4, "Weight buffer should be 4D")
        image_shape[0] = roundup_div4(shape[1]) * shape[2] * shape[3]
        image_shape[1] = shape[0]
    else:
        mace_check(False, "OpenCL Image do not support type "
                   + str(buffer_type))
    return image_shape
