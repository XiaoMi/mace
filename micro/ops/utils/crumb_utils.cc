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

#include "micro/ops/utils/crumb_utils.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"

namespace micro {
namespace ops {
namespace crumb {

MaceStatus ComputeBias(const mifloat *input, const int32_t *input_dims,
                       const uint32_t input_dim_size, const mifloat *bias,
                       const int32_t channel, mifloat *output) {
  MACE_ASSERT(input != NULL && input_dims != NULL && input_dim_size > 0
                  && bias != NULL && channel > 0 && output != NULL);
  const int32_t outer_size =
      base::accumulate_multi(input_dims, 0, input_dim_size - 1);
  for (int32_t i = 0; i < outer_size; ++i) {
    const int32_t outer_base = i * channel;
    for (int32_t c = 0; c < channel; ++c) {
      const int32_t idx = outer_base + c;
      output[idx] = input[idx] + bias[c];
    }
  }
  return MACE_SUCCESS;
}

}  // namespace crumb
}  // namespace ops
}  // namespace micro
