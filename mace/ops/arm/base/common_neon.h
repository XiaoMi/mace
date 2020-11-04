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

#ifndef MACE_OPS_ARM_BASE_COMMON_NEON_H_
#define MACE_OPS_ARM_BASE_COMMON_NEON_H_

#include <arm_neon.h>

#include "mace/core/bfloat16.h"

namespace mace {
namespace ops {
namespace arm {

typedef struct float32x8_t {
  float32x4_t val[2];
} float32x8_t;

#if !defined(__aarch64__)
inline float vaddvq_f32(float32x4_t v) {
  float32x2_t _sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
  _sum = vpadd_f32(_sum, _sum);
  return vget_lane_f32(_sum, 0);
}
#endif

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

inline void neon_vec_left_shift_1(const float32x4_t &src,
                                  float32x4_t *dst) {
  (*dst)[0] = src[1];
  (*dst)[1] = src[2];
  (*dst)[2] = src[3];
}

inline void neon_vec_left_shift_2(const float32x4_t &src,
                                  float32x4_t *dst) {
  (*dst)[0] = src[2];
  (*dst)[1] = src[3];
}

inline void neon_vec_left_shift_3(const float32x4_t &src,
                                  float32x4_t *dst) {
  (*dst)[0] = src[3];
}

inline void neon_vec_right_shift_1(const float32x4_t &src,
                                   float32x4_t *dst) {
  (*dst)[1] = src[0];
  (*dst)[2] = src[1];
  (*dst)[3] = src[2];
}

inline void neon_vec_right_shift_2(const float32x4_t &src,
                                   float32x4_t *dst) {
  (*dst)[2] = src[0];
  (*dst)[3] = src[1];
}

inline void neon_vec_right_shift_3(const float32x4_t &src,
                                   float32x4_t *dst) {
  (*dst)[3] = src[0];
}

inline float32x2_t vld1(const float *ptr) {
  return vld1_f32(ptr);
}

inline void vst1(float *ptr, float32x2_t v) {
  vst1_f32(ptr, v);
}

inline float32x4_t vld1q(const float *ptr) {
  return vld1q_f32(ptr);
}

inline float32x4x2_t vld2q(const float *ptr) {
  return vld2q_f32(ptr);
}

inline float32x4x3_t vld3q(const float *ptr) {
  return vld3q_f32(ptr);
}

inline void vst1q(float *ptr, float32x4_t v) {
  vst1q_f32(ptr, v);
}

inline void vst2q(float *ptr, float32x4x2_t v) {
  vst2q_f32(ptr, v);
}

inline void vst3q(float *ptr, float32x4x3_t v) {
  vst3q_f32(ptr, v);
}

inline float32x8_t vld1o(float *ptr) {
  return {{vld1q_f32(ptr), vld1q_f32(ptr + 4)}};
}

inline void vst1o(float *ptr, float32x8_t v) {
  vst1q_f32(ptr, v.val[0]);
  vst1q_f32(ptr + 4, v.val[1]);
}

#if defined(MACE_ENABLE_AMR82)

// load of 4D vector
inline float16x4_t vld1(const float16_t *ptr) {
  return vld1_fp16(ptr);
}

// store of 4D vector
inline void vst1(float16_t *ptr, float16x4_t v) {
  vst1_fp16(ptr, v);
}

// load of 8D vector
inline float16x8_t vld1q(const float16_t *ptr) {
  return vld1q_fp16(ptr);
}

// load of 2 8D vectors and perform de-interleaving
inline float16x8x2_t vld2q(const float16_t *ptr) {
  return vld2q_fp16(ptr);
}

// store of 8D vector
inline void vst1q(float16_t *ptr, const float16x8_t v) {
  vst1q_fp16(ptr, v);
}

// store of 2 8D vectors and perform interleaving
inline void vst2q(float16_t *ptr, const float16x8x2_t v) {
  vst2q_fp16(ptr, v);
}

#endif  // MACE_ENABLE_FP16

#if defined(MACE_ENABLE_BFLOAT16)

// load of 2D vector
inline float32x2_t vld1_bf16(const BFloat16 *ptr) {
  return (float32x2_t){ptr[0], ptr[1]};  // NOLINT(readability/braces)
}

inline float32x2_t vld1_bf16(const uint16_t *ptr) {
  return vld1_bf16(reinterpret_cast<const BFloat16 *>(ptr));
}

inline float32x2_t vld1(const BFloat16 *ptr) {
  return vld1_bf16(ptr);
}

inline float32x2_t vld1(const uint16_t *ptr) {
  return vld1_bf16(reinterpret_cast<const BFloat16 *>(ptr));
}

// store of 2D vector
inline void vst1_bf16(BFloat16 *ptr, float32x2_t v) {
  ptr[0] = v[0];
  ptr[1] = v[1];
}

inline void vst1_bf16(uint16_t *ptr, float32x2_t v) {
  vst1_bf16(reinterpret_cast<BFloat16 *>(ptr), v);
}

inline void vst1(BFloat16 *ptr, float32x2_t v) {
  vst1_bf16(ptr, v);
}

inline void vst1(uint16_t *ptr, float32x2_t v) {
  vst1_bf16(reinterpret_cast<BFloat16 *>(ptr), v);
}

// load of 4D vector
inline float32x4_t vld1q_bf16(const uint16_t *ptr) {
  return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
}

inline float32x4_t vld1q_bf16(const BFloat16 *ptr) {
  return vld1q_bf16(reinterpret_cast<const uint16_t *>(ptr));
}

inline float32x4_t vld1q(const uint16_t *ptr) {
  return vld1q_bf16(ptr);
}

inline float32x4_t vld1q(const BFloat16 *ptr) {
  return vld1q_bf16(reinterpret_cast<const uint16_t *>(ptr));
}

// load of 2 4D vectors and perform de-interleaving
inline float32x4x2_t vld2q_bf16(const uint16_t *ptr) {
  uint16x4x2_t u = vld2_u16(ptr);
  return {{vreinterpretq_f32_u32(vshll_n_u16(u.val[0], 16)),
           vreinterpretq_f32_u32(vshll_n_u16(u.val[1], 16))}};
}

inline float32x4x2_t vld2q_bf16(const BFloat16 *ptr) {
  return vld2q_bf16(reinterpret_cast<const uint16_t *>(ptr));
}

inline float32x4x2_t vld2q(const uint16_t *ptr) {
  return vld2q_bf16(ptr);
}

inline float32x4x2_t vld2q(const BFloat16 *ptr) {
  return vld2q_bf16(reinterpret_cast<const uint16_t *>(ptr));
}

// load of 3 4D vectors and perform de-interleaving
inline float32x4x3_t vld3q_bf16(const uint16_t *ptr) {
  uint16x4x3_t u = vld3_u16(ptr);
  return {{vreinterpretq_f32_u32(vshll_n_u16(u.val[0], 16)),
           vreinterpretq_f32_u32(vshll_n_u16(u.val[1], 16)),
           vreinterpretq_f32_u32(vshll_n_u16(u.val[2], 16))}};
}

inline float32x4x3_t vld3q_bf16(const BFloat16 *ptr) {
  return vld3q_bf16(reinterpret_cast<const uint16_t *>(ptr));
}

inline float32x4x3_t vld3q(const uint16_t *ptr) {
  return vld3q_bf16(ptr);
}

inline float32x4x3_t vld3q(const BFloat16 *ptr) {
  return vld3q_bf16(reinterpret_cast<const uint16_t *>(ptr));
}

// store of 4D vector
inline void vst1q_bf16(uint16_t *ptr, const float32x4_t v) {
  vst1_u16(ptr, vshrn_n_u32(vreinterpretq_u32_f32(v), 16));
}

inline void vst1q_bf16(BFloat16 *ptr, const float32x4_t v) {
  vst1q_bf16(reinterpret_cast<uint16_t *>(ptr), v);
}

inline void vst1q(uint16_t *ptr, const float32x4_t v) {
  vst1q_bf16(ptr, v);
}

inline void vst1q(BFloat16 *ptr, const float32x4_t v) {
  vst1q_bf16(reinterpret_cast<uint16_t *>(ptr), v);
}

// store of 2 4D vectors and perform interleaving
inline void vst2q_bf16(uint16_t *ptr, const float32x4x2_t v) {
  uint16x4x2_t u = {{vshrn_n_u32(vreinterpretq_u32_f32(v.val[0]), 16),
                     vshrn_n_u32(vreinterpretq_u32_f32(v.val[1]), 16)}};
  vst2_u16(ptr, u);
}

inline void vst2q_bf16(BFloat16 *ptr, const float32x4x2_t v) {
  vst2q_bf16(reinterpret_cast<uint16_t *>(ptr), v);
}

inline void vst2q(uint16_t *ptr, const float32x4x2_t v) {
  vst2q_bf16(ptr, v);
}

inline void vst2q(BFloat16 *ptr, const float32x4x2_t v) {
  vst2q_bf16(reinterpret_cast<uint16_t *>(ptr), v);
}

// store of 3 4D vectors and perform interleaving
inline void vst3q_bf16(uint16_t *ptr, const float32x4x3_t v) {
  uint16x4x3_t u = {{vshrn_n_u32(vreinterpretq_u32_f32(v.val[0]), 16),
                    vshrn_n_u32(vreinterpretq_u32_f32(v.val[0]), 16),
                    vshrn_n_u32(vreinterpretq_u32_f32(v.val[0]), 16)}};
  vst3_u16(ptr, u);
}

inline void vst3q_bf16(BFloat16 *ptr, const float32x4x3_t v) {
  vst3q_bf16(reinterpret_cast<uint16_t *>(ptr), v);
}

inline void vst3q(uint16_t *ptr, const float32x4x3_t v) {
  vst3q_bf16(ptr, v);
}

inline void vst3q(BFloat16 *ptr, const float32x4x3_t v) {
  vst3q_bf16(reinterpret_cast<uint16_t *>(ptr), v);
}

// load of 8D vector
inline float32x8_t vld1o_bf16(const uint16_t *ptr) {
  uint16x8_t u = vld1q_u16(ptr);
  return {{vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u), 16)),
           vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(u), 16))}};
}

inline float32x8_t vld1o_bf16(const BFloat16 *ptr) {
  return vld1o_bf16(reinterpret_cast<const uint16_t *>(ptr));
}

inline float32x8_t vld1o(const uint16_t *ptr) {
  return vld1o_bf16(ptr);
}

inline float32x8_t vld1o(const BFloat16 *ptr) {
  return vld1o_bf16(reinterpret_cast<const uint16_t *>(ptr));
}

// store of 8D vector
inline void vst1o_bf16(uint16_t *ptr, const float32x8_t v) {
  vst1q_u16(ptr, vcombine_u16(
      vshrn_n_u32(vreinterpretq_u32_f32(v.val[0]), 16),
      vshrn_n_u32(vreinterpretq_u32_f32(v.val[1]), 16)));
}

inline void vst1o_bf16(BFloat16 *ptr, const float32x8_t v) {
  vst1o_bf16(reinterpret_cast<uint16_t *>(ptr), v);
}

inline void vst1o(uint16_t *ptr, const float32x8_t v) {
  vst1o_bf16(ptr, v);
}

inline void vst1o(BFloat16 *ptr, const float32x8_t v) {
  vst1o_bf16(reinterpret_cast<uint16_t *>(ptr), v);
}

#endif  // MACE_ENABLE_BFLOAT16

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_COMMON_NEON_H_
