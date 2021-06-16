// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_COMMON_UTILS_H_
#define MACE_OPS_COMMON_UTILS_H_

#include <set>
#include <string>

#include "mace/core/types.h"
#include "mace/ops/common/activation_type.h"

namespace mace {

class Tensor;

namespace ops {
namespace common {
namespace utils {

constexpr int64_t kTableSize = (1u << 10);

inline float CalculateResizeScale(index_t in_size,
                                  index_t out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
         ? (in_size - 1) / static_cast<float>(out_size - 1)
         : in_size / static_cast<float>(out_size);
}

void GetSizeParamFromTensor(const Tensor *size_tensor, index_t *out_height,
                            index_t *out_width);

void FillBuiltOptions(std::set<std::string> *built_options,
                            const ActivationType &activation);
}  // namespace utils
}  // namespace common
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_UTILS_H_
