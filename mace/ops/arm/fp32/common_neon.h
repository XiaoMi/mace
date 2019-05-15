// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_ARM_FP32_COMMON_NEON_H_
#define MACE_OPS_ARM_FP32_COMMON_NEON_H_

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

inline float32x4_t neon_vfma_lane_0(float32x4_t a,
                          float32x4_t b,
                          float32x4_t c) {
#ifdef __aarch64__
  return vfmaq_laneq_f32(a, b, c, 0);
#else
  return vmlaq_lane_f32(a, b, vget_low_f32(c), 0);
#endif
}

inline float32x4_t neon_vfma_lane_1(float32x4_t a,
                          float32x4_t b,
                          float32x4_t c) {
#ifdef __aarch64__
  return vfmaq_laneq_f32(a, b, c, 1);
#else
  return vmlaq_lane_f32(a, b, vget_low_f32(c), 1);
#endif
}

inline float32x4_t neon_vfma_lane_2(float32x4_t a,
                          float32x4_t b,
                          float32x4_t c) {
#ifdef __aarch64__
  return vfmaq_laneq_f32(a, b, c, 2);
#else
  return vmlaq_lane_f32(a, b, vget_high_f32(c), 0);
#endif
}

inline float32x4_t neon_vfma_lane_3(float32x4_t a,
                          float32x4_t b,
                          float32x4_t c) {
#ifdef __aarch64__
  return vfmaq_laneq_f32(a, b, c, 3);
#else
  return vmlaq_lane_f32(a, b, vget_high_f32(c), 1);
#endif
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
#endif  // MACE_ENABLE_NEON

#endif  // MACE_OPS_ARM_FP32_COMMON_NEON_H_
