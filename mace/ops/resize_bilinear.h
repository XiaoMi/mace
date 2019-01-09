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

#ifndef MACE_OPS_RESIZE_BILINEAR_H_
#define MACE_OPS_RESIZE_BILINEAR_H_

#include "mace/core/types.h"

namespace mace {
namespace ops {
namespace resize_bilinear {
inline float CalculateResizeScale(index_t in_size,
                                  index_t out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
         ? (in_size - 1) / static_cast<float>(out_size - 1)
         : in_size / static_cast<float>(out_size);
}
}  // namespace resize_bilinear
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_RESIZE_BILINEAR_H_
