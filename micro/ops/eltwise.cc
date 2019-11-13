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

#include "micro/ops/eltwise.h"

#include "micro/base/logging.h"

namespace micro {
namespace ops {
namespace eltwise {
bool ShapeIsEqual(const int32_t *dims0,
                  const int32_t *dims1, uint32_t dim_size) {
  while (--dim_size > 0) {
    if (dims0[dim_size] != dims1[dim_size])
      return false;
  }
  return true;
}

int32_t GetIndex(const int32_t *shape,
                 const int32_t *index, int32_t dim_size) {
  int32_t idx = 0;
  for (int32_t i = 0; i < dim_size; ++i) {
    if (shape[i] > 1) {
      idx = idx * shape[i] + index[i];
    }
  }
  return idx;
}

void IncreaseIndex(const int32_t *shape, int32_t **index, int32_t dim_size) {
  for (int32_t i = dim_size - 1; i >= 0; --i) {
    ++(*index)[i];
    if ((*index)[i] >= shape[i]) {
      (*index)[i] -= shape[i];
    } else {
      break;
    }
  }
}
}  // namespace eltwise
}  // namespace ops
}  // namespace micro
