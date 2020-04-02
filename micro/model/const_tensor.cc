// Copyright 2020 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "micro/model/const_tensor.h"

namespace micro {
namespace model {

MACE_DEFINE_ARRAY_FUNC(ConstTensor, int32_t, dim, dims_)
MACE_DEFINE_OBJECT_FUNC(ConstTensor, DataType, data_type)
MACE_DEFINE_ARRAY_FUNC(ConstTensor, float, float_data, float_datas_)
MACE_DEFINE_ARRAY_FUNC(ConstTensor, int32_t, int32_data, int32_datas_)
MACE_DEFINE_STRING_FUNC(ConstTensor, name, name_)
MACE_DEFINE_OBJECT_FUNC(ConstTensor, int32_t, offset)
MACE_DEFINE_OBJECT_FUNC(ConstTensor, int32_t, data_size)
MACE_DEFINE_OBJECT_FUNC(ConstTensor, float, scale)
MACE_DEFINE_OBJECT_FUNC(ConstTensor, int32_t, zero_point)
MACE_DEFINE_OBJECT_FUNC(ConstTensor, float, minval)
MACE_DEFINE_OBJECT_FUNC(ConstTensor, float, maxval)
MACE_DEFINE_OBJECT_FUNC(ConstTensor, bool, quantized)
MACE_DEFINE_OBJECT_FUNC(ConstTensor, uint32_t, node_id)

const int32_t *ConstTensor::dim() const {
  const int32_t *array = reinterpret_cast<const int32_t *>(
      reinterpret_cast<const uint8_t *>(this) + dims_.offset_);
  return array;
}

}  // namespace model
}  // namespace micro
