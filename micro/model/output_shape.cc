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

#include "micro/model/output_shape.h"

namespace micro {
namespace model {

MACE_DEFINE_ARRAY_FUNC(OutputShape, int32_t, dim, dims_)

const int32_t *OutputShape::dim() const {
  const int32_t *array = reinterpret_cast<const int32_t *>(
      reinterpret_cast<const char *>(this) + dims_.offset_);
  return array;
}

int32_t *OutputShape::mutable_dim() {
  char *base_addr = reinterpret_cast<char *>(const_cast<OutputShape *>(this));
  int32_t *array = reinterpret_cast<int32_t *>(base_addr + dims_.offset_);
  return array;
}

}  // namespace model
}  // namespace micro
