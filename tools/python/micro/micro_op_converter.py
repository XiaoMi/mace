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

from transform.base_converter import ConverterUtil
from transform.base_converter import DataFormat
from transform.base_converter import MaceKeyword
from transform.base_converter import MaceOp
from utils.util import mace_check
import numpy as np


class MicroOpConverter:
    def __init__(self, pb_model, model_weights, data_type=np.float32):
        self.net_def = pb_model
        self.model_weights = model_weights
        self.weight_bytes = bytearray(model_weights)
        self.data_type = data_type
        self._consts = {}
        for tensor in self.net_def.tensors:
            self._consts[tensor.name] = tensor

    def convert_filters_format(self):
        arg_format = ConverterUtil.get_arg(self.net_def,
                                           MaceKeyword.mace_filter_format_str)
        mace_check(arg_format.i == DataFormat.OIHW.value, "Invalid model")
        arg_format.i = DataFormat.OHWI.value

        transposed_filter = set()
        for op in self.net_def.op:
            # OIHW => OHWI
            if (op.type == MaceOp.Conv2D.name or
                op.type == MaceOp.DepthwiseConv2d.name) and \
                    op.input[1] not in transposed_filter:
                print("transform filter: %s" % op.type)
                filter = self._consts[op.input[1]]
                tensor_data = np.frombuffer(self.weight_bytes, self.data_type,
                                            filter.data_size, filter.offset)
                filter_data = np.array(tensor_data).reshape(filter.dims) \
                    .transpose(0, 2, 3, 1)
                filter_bytes = np.array(filter_data).tobytes()
                slice_end = filter.offset + len(filter_bytes)
                self.model_weights[filter.offset: slice_end] = filter_bytes
                filter.dims[:] = filter_data.shape
                transposed_filter.add(op.input[1])

    def convert_op_params(self):
        self.convert_filters_format()
