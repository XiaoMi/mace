// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_KERNELS_FIXPOINT_H_
#define MACE_KERNELS_FIXPOINT_H_

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include "mace/core/types.h"

namespace mace {
namespace kernels {

inline uint8_t FindMax(const uint8_t *xs, const index_t size) {
  uint8_t max_value = 0;
  index_t i = 0;
#if defined(MACE_ENABLE_NEON)
  uint8x16_t vmax16_0 = vdupq_n_u8(0);
  uint8x16_t vmax16_1 = vdupq_n_u8(0);
  for (; i <= size - 32; i += 32) {
    vmax16_0 = vmaxq_u8(vmax16_0, vld1q_u8(xs + i + 0));
    vmax16_1 = vmaxq_u8(vmax16_1, vld1q_u8(xs + i + 16));
  }
  uint8x16_t vmax16 = vmaxq_u8(vmax16_0, vmax16_1);
  if (i <= size - 16) {
    vmax16 = vmaxq_u8(vmax16, vld1q_u8(xs + i));
    i += 16;
  }
  uint8x8_t vmax8 = vmax_u8(vget_low_u8(vmax16), vget_high_u8(vmax16));
  if (i <= size - 8) {
    vmax8 = vmax_u8(vmax8, vld1_u8(xs + i));
    i += 8;
  }
  uint8x8_t vmax4 = vmax_u8(vmax8, vext_u8(vmax8, vmax8, 4));
  uint8x8_t vmax2 = vmax_u8(vmax4, vext_u8(vmax4, vmax4, 2));
  uint8x8_t vmax1 = vpmax_u8(vmax2, vmax2);
  max_value = vget_lane_u8(vmax1, 0);
#endif
  for (; i < size; ++i) {
    max_value = std::max(max_value, xs[i]);
  }
  return max_value;
}


}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_FIXPOINT_H_

