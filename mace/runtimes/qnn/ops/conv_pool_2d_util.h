// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_RUNTIMES_QNN_OPS_CONV_POOL_2D_UTIL_H_
#define MACE_RUNTIMES_QNN_OPS_CONV_POOL_2D_UTIL_H_

#include <vector>

#include "mace/core/types.h"

namespace mace {
void CalcPadding(const uint32_t *input_shape,
                 const uint32_t *filter_shape,
                 const uint32_t *output_shape,
                 const int *dilations,
                 const int *strides,
                 const std::vector<int> &padding_values,
                 std::vector<uint32_t> *qnn_paddings);
}  // namespace mace

#endif  // MACE_RUNTIMES_QNN_OPS_CONV_POOL_2D_UTIL_H_
