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

#ifndef MACE_KERNELS_ARM_DECONV_2D_NEON_H_
#define MACE_KERNELS_ARM_DECONV_2D_NEON_H_

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include "mace/core/types.h"

namespace mace {
namespace kernels {

void Deconv2dNeonK3x3S1(const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *out_shape,
                        float *output);

void Deconv2dNeonK3x3S2(const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *out_shape,
                        float *output);

void Deconv2dNeonK4x4S1(const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *out_shape,
                        float *output);

void Deconv2dNeonK4x4S2(const float *input,
                        const float *filter,
                        const index_t *in_shape,
                        const index_t *out_shape,
                        float *output);

#ifdef MACE_ENABLE_NEON
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
#endif

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_ARM_DECONV_2D_NEON_H_
