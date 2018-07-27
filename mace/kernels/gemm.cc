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

#include <algorithm>
#include <cstring>

#include "mace/core/tensor.h"
#include "mace/kernels/gemm.h"

/**
 * Gemm does fast matrix multiplications with batch.
 * It is optimized for arm64-v8 and armeabi-v7a using neon.
 *
 * We adopt two-level tiling to make better use of l1 cache and register.
 * For register tiling, function like GemmXYZ computes gemm for
 * matrix[X, Y] * matrix[Y, Z] with all data being able to fit in register.
 * For cache tiling, we try to compute one block of multiplication with
 * two input matrices and one output matrix fit in l1 cache.
 */

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#if defined(MACE_ENABLE_NEON) && !defined(__aarch64__)
#define vaddvq_f32(v) ((v)[0] + (v)[1] + (v)[2] + (v)[3])
#endif

namespace mace {
namespace kernels {

namespace {
inline void GemmBlock(const float *A,
                      const float *B,
                      const index_t height,
                      const index_t K,
                      const index_t width,
                      const index_t stride_a,
                      const index_t stride_b,
                      const index_t stride_c,
                      float *C) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * stride_c + j] += A[i * stride_a + k] * B[k * stride_b + j];
      }
    }
  }
}

#if defined(MACE_ENABLE_NEON)
#if defined(__aarch64__)
#define MACE_GEMM_PART_CAL_8(RC, RA, RAN)        \
  c##RC = vfmaq_laneq_f32(c##RC, b0, a##RA, 0);  \
  c##RC = vfmaq_laneq_f32(c##RC, b1, a##RA, 1);  \
  c##RC = vfmaq_laneq_f32(c##RC, b2, a##RA, 2);  \
  c##RC = vfmaq_laneq_f32(c##RC, b3, a##RA, 3);  \
  c##RC = vfmaq_laneq_f32(c##RC, b4, a##RAN, 0); \
  c##RC = vfmaq_laneq_f32(c##RC, b5, a##RAN, 1); \
  c##RC = vfmaq_laneq_f32(c##RC, b6, a##RAN, 2); \
  c##RC = vfmaq_laneq_f32(c##RC, b7, a##RAN, 3);
#else
#define MACE_GEMM_PART_CAL_8(RC, RA, RAN)                      \
  c##RC = vmlaq_lane_f32(c##RC, b0, vget_low_f32(a##RA), 0);   \
  c##RC = vmlaq_lane_f32(c##RC, b1, vget_low_f32(a##RA), 1);   \
  c##RC = vmlaq_lane_f32(c##RC, b2, vget_high_f32(a##RA), 0);  \
  c##RC = vmlaq_lane_f32(c##RC, b3, vget_high_f32(a##RA), 1);  \
  c##RC = vmlaq_lane_f32(c##RC, b4, vget_low_f32(a##RAN), 0);  \
  c##RC = vmlaq_lane_f32(c##RC, b5, vget_low_f32(a##RAN), 1);  \
  c##RC = vmlaq_lane_f32(c##RC, b6, vget_high_f32(a##RAN), 0); \
  c##RC = vmlaq_lane_f32(c##RC, b7, vget_high_f32(a##RAN), 1);
#endif
#endif

#if defined(MACE_ENABLE_NEON)
#if defined(__aarch64__)
#define MACE_GEMM_PART_CAL_4(RC)                \
  c##RC = vfmaq_laneq_f32(c##RC, b0, a##RC, 0); \
  c##RC = vfmaq_laneq_f32(c##RC, b1, a##RC, 1); \
  c##RC = vfmaq_laneq_f32(c##RC, b2, a##RC, 2); \
  c##RC = vfmaq_laneq_f32(c##RC, b3, a##RC, 3);
#else
#define MACE_GEMM_PART_CAL_4(RC)                              \
  c##RC = vmlaq_lane_f32(c##RC, b0, vget_low_f32(a##RC), 0);  \
  c##RC = vmlaq_lane_f32(c##RC, b1, vget_low_f32(a##RC), 1);  \
  c##RC = vmlaq_lane_f32(c##RC, b2, vget_high_f32(a##RC), 0); \
  c##RC = vmlaq_lane_f32(c##RC, b3, vget_high_f32(a##RC), 1);
#endif
#endif

inline void Gemm144(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  MACE_UNUSED(stride_a);
  MACE_UNUSED(stride_c);
  float32x4_t a0;
  float32x4_t b0, b1, b2, b3;
  float32x4_t c0;

  a0 = vld1q_f32(a_ptr);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);

  c0 = vld1q_f32(c_ptr);

  MACE_GEMM_PART_CAL_4(0);

  vst1q_f32(c_ptr, c0);
#else
  GemmBlock(a_ptr, b_ptr, 1, 4, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm244(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1;
  float32x4_t b0, b1, b2, b3;
  float32x4_t c0, c1;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 1 * stride_a);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);

  MACE_GEMM_PART_CAL_4(0);
  MACE_GEMM_PART_CAL_4(1);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
#else
  GemmBlock(a_ptr, b_ptr, 2, 4, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm344(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2;
  float32x4_t b0, b1, b2, b3;
  float32x4_t c0, c1, c2;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 1 * stride_a);
  a2 = vld1q_f32(a_ptr + 2 * stride_a);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);
  c2 = vld1q_f32(c_ptr + 2 * stride_c);

  MACE_GEMM_PART_CAL_4(0);
  MACE_GEMM_PART_CAL_4(1);
  MACE_GEMM_PART_CAL_4(2);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
  vst1q_f32(c_ptr + 2 * stride_c, c2);
#else
  GemmBlock(a_ptr, b_ptr, 3, 4, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm444(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2, a3;
  float32x4_t b0, b1, b2, b3;
  float32x4_t c0, c1, c2, c3;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 1 * stride_a);
  a2 = vld1q_f32(a_ptr + 2 * stride_a);
  a3 = vld1q_f32(a_ptr + 3 * stride_a);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);
  c2 = vld1q_f32(c_ptr + 2 * stride_c);
  c3 = vld1q_f32(c_ptr + 3 * stride_c);

  MACE_GEMM_PART_CAL_4(0);
  MACE_GEMM_PART_CAL_4(1);
  MACE_GEMM_PART_CAL_4(2);
  MACE_GEMM_PART_CAL_4(3);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
  vst1q_f32(c_ptr + 2 * stride_c, c2);
  vst1q_f32(c_ptr + 3 * stride_c, c3);
#else
  GemmBlock(a_ptr, b_ptr, 4, 4, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm544(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2, a3, a4;
  float32x4_t b0, b1, b2, b3;
  float32x4_t c0, c1, c2, c3, c4;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 1 * stride_a);
  a2 = vld1q_f32(a_ptr + 2 * stride_a);
  a3 = vld1q_f32(a_ptr + 3 * stride_a);
  a4 = vld1q_f32(a_ptr + 4 * stride_a);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);
  c2 = vld1q_f32(c_ptr + 2 * stride_c);
  c3 = vld1q_f32(c_ptr + 3 * stride_c);
  c4 = vld1q_f32(c_ptr + 4 * stride_c);

  MACE_GEMM_PART_CAL_4(0);
  MACE_GEMM_PART_CAL_4(1);
  MACE_GEMM_PART_CAL_4(2);
  MACE_GEMM_PART_CAL_4(3);
  MACE_GEMM_PART_CAL_4(4);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
  vst1q_f32(c_ptr + 2 * stride_c, c2);
  vst1q_f32(c_ptr + 3 * stride_c, c3);
  vst1q_f32(c_ptr + 4 * stride_c, c4);
#else
  GemmBlock(a_ptr, b_ptr, 5, 4, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm644(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2, a3, a4, a5;
  float32x4_t b0, b1, b2, b3;
  float32x4_t c0, c1, c2, c3, c4, c5;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 1 * stride_a);
  a2 = vld1q_f32(a_ptr + 2 * stride_a);
  a3 = vld1q_f32(a_ptr + 3 * stride_a);
  a4 = vld1q_f32(a_ptr + 4 * stride_a);
  a5 = vld1q_f32(a_ptr + 5 * stride_a);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);
  c2 = vld1q_f32(c_ptr + 2 * stride_c);
  c3 = vld1q_f32(c_ptr + 3 * stride_c);
  c4 = vld1q_f32(c_ptr + 4 * stride_c);
  c5 = vld1q_f32(c_ptr + 5 * stride_c);

  MACE_GEMM_PART_CAL_4(0);
  MACE_GEMM_PART_CAL_4(1);
  MACE_GEMM_PART_CAL_4(2);
  MACE_GEMM_PART_CAL_4(3);
  MACE_GEMM_PART_CAL_4(4);
  MACE_GEMM_PART_CAL_4(5);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
  vst1q_f32(c_ptr + 2 * stride_c, c2);
  vst1q_f32(c_ptr + 3 * stride_c, c3);
  vst1q_f32(c_ptr + 4 * stride_c, c4);
  vst1q_f32(c_ptr + 5 * stride_c, c5);
#else
  GemmBlock(a_ptr, b_ptr, 6, 4, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void GemmX44(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr,
                    int row) {
  switch (row) {
    case 1:
      Gemm144(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 2:
      Gemm244(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 3:
      Gemm344(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 4:
      Gemm444(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 5:
      Gemm544(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 6:
      Gemm644(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    default:
      MACE_NOT_IMPLEMENTED;
  }
}

inline void Gemm884(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
      a15;
  float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
  float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 4);
  a2 = vld1q_f32(a_ptr + 1 * stride_a);
  a3 = vld1q_f32(a_ptr + 1 * stride_a + 4);
  a4 = vld1q_f32(a_ptr + 2 * stride_a);
  a5 = vld1q_f32(a_ptr + 2 * stride_a + 4);
  a6 = vld1q_f32(a_ptr + 3 * stride_a);
  a7 = vld1q_f32(a_ptr + 3 * stride_a + 4);
  a8 = vld1q_f32(a_ptr + 4 * stride_a);
  a9 = vld1q_f32(a_ptr + 4 * stride_a + 4);
  a10 = vld1q_f32(a_ptr + 5 * stride_a);
  a11 = vld1q_f32(a_ptr + 5 * stride_a + 4);
  a12 = vld1q_f32(a_ptr + 6 * stride_a);
  a13 = vld1q_f32(a_ptr + 6 * stride_a + 4);
  a14 = vld1q_f32(a_ptr + 7 * stride_a);
  a15 = vld1q_f32(a_ptr + 7 * stride_a + 4);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);
  b4 = vld1q_f32(b_ptr + 4 * stride_b);
  b5 = vld1q_f32(b_ptr + 5 * stride_b);
  b6 = vld1q_f32(b_ptr + 6 * stride_b);
  b7 = vld1q_f32(b_ptr + 7 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);
  c2 = vld1q_f32(c_ptr + 2 * stride_c);
  c3 = vld1q_f32(c_ptr + 3 * stride_c);
  c4 = vld1q_f32(c_ptr + 4 * stride_c);
  c5 = vld1q_f32(c_ptr + 5 * stride_c);
  c6 = vld1q_f32(c_ptr + 6 * stride_c);
  c7 = vld1q_f32(c_ptr + 7 * stride_c);

  MACE_GEMM_PART_CAL_8(0, 0, 1);
  MACE_GEMM_PART_CAL_8(1, 2, 3);
  MACE_GEMM_PART_CAL_8(2, 4, 5);
  MACE_GEMM_PART_CAL_8(3, 6, 7);
  MACE_GEMM_PART_CAL_8(4, 8, 9);
  MACE_GEMM_PART_CAL_8(5, 10, 11);
  MACE_GEMM_PART_CAL_8(6, 12, 13);
  MACE_GEMM_PART_CAL_8(7, 14, 15);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
  vst1q_f32(c_ptr + 2 * stride_c, c2);
  vst1q_f32(c_ptr + 3 * stride_c, c3);
  vst1q_f32(c_ptr + 4 * stride_c, c4);
  vst1q_f32(c_ptr + 5 * stride_c, c5);
  vst1q_f32(c_ptr + 6 * stride_c, c6);
  vst1q_f32(c_ptr + 7 * stride_c, c7);
#else
  GemmBlock(a_ptr, b_ptr, 8, 8, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm184(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  MACE_UNUSED(stride_a);
  MACE_UNUSED(stride_c);

  float32x4_t a0, a1;
  float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
  float32x4_t c0;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 4);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);
  b4 = vld1q_f32(b_ptr + 4 * stride_b);
  b5 = vld1q_f32(b_ptr + 5 * stride_b);
  b6 = vld1q_f32(b_ptr + 6 * stride_b);
  b7 = vld1q_f32(b_ptr + 7 * stride_b);

  c0 = vld1q_f32(c_ptr);

  MACE_GEMM_PART_CAL_8(0, 0, 1);

  vst1q_f32(c_ptr, c0);
#else
  GemmBlock(a_ptr, b_ptr, 1, 8, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm284(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2, a3;
  float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
  float32x4_t c0, c1;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 4);
  a2 = vld1q_f32(a_ptr + 1 * stride_a);
  a3 = vld1q_f32(a_ptr + 1 * stride_a + 4);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);
  b4 = vld1q_f32(b_ptr + 4 * stride_b);
  b5 = vld1q_f32(b_ptr + 5 * stride_b);
  b6 = vld1q_f32(b_ptr + 6 * stride_b);
  b7 = vld1q_f32(b_ptr + 7 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);

  MACE_GEMM_PART_CAL_8(0, 0, 1);
  MACE_GEMM_PART_CAL_8(1, 2, 3);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
#else
  GemmBlock(a_ptr, b_ptr, 2, 8, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm384(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2, a3, a4, a5;
  float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
  float32x4_t c0, c1, c2;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 4);
  a2 = vld1q_f32(a_ptr + 1 * stride_a);
  a3 = vld1q_f32(a_ptr + 1 * stride_a + 4);
  a4 = vld1q_f32(a_ptr + 2 * stride_a);
  a5 = vld1q_f32(a_ptr + 2 * stride_a + 4);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);
  b4 = vld1q_f32(b_ptr + 4 * stride_b);
  b5 = vld1q_f32(b_ptr + 5 * stride_b);
  b6 = vld1q_f32(b_ptr + 6 * stride_b);
  b7 = vld1q_f32(b_ptr + 7 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);
  c2 = vld1q_f32(c_ptr + 2 * stride_c);

  MACE_GEMM_PART_CAL_8(0, 0, 1);
  MACE_GEMM_PART_CAL_8(1, 2, 3);
  MACE_GEMM_PART_CAL_8(2, 4, 5);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
  vst1q_f32(c_ptr + 2 * stride_c, c2);
#else
  GemmBlock(a_ptr, b_ptr, 3, 8, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm484(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2, a3, a4, a5, a6, a7;
  float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
  float32x4_t c0, c1, c2, c3;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 4);
  a2 = vld1q_f32(a_ptr + 1 * stride_a);
  a3 = vld1q_f32(a_ptr + 1 * stride_a + 4);
  a4 = vld1q_f32(a_ptr + 2 * stride_a);
  a5 = vld1q_f32(a_ptr + 2 * stride_a + 4);
  a6 = vld1q_f32(a_ptr + 3 * stride_a);
  a7 = vld1q_f32(a_ptr + 3 * stride_a + 4);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);
  b4 = vld1q_f32(b_ptr + 4 * stride_b);
  b5 = vld1q_f32(b_ptr + 5 * stride_b);
  b6 = vld1q_f32(b_ptr + 6 * stride_b);
  b7 = vld1q_f32(b_ptr + 7 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);
  c2 = vld1q_f32(c_ptr + 2 * stride_c);
  c3 = vld1q_f32(c_ptr + 3 * stride_c);

  MACE_GEMM_PART_CAL_8(0, 0, 1);
  MACE_GEMM_PART_CAL_8(1, 2, 3);
  MACE_GEMM_PART_CAL_8(2, 4, 5);
  MACE_GEMM_PART_CAL_8(3, 6, 7);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
  vst1q_f32(c_ptr + 2 * stride_c, c2);
  vst1q_f32(c_ptr + 3 * stride_c, c3);
#else
  GemmBlock(a_ptr, b_ptr, 4, 8, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm584(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;
  float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
  float32x4_t c0, c1, c2, c3, c4;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 4);
  a2 = vld1q_f32(a_ptr + 1 * stride_a);
  a3 = vld1q_f32(a_ptr + 1 * stride_a + 4);
  a4 = vld1q_f32(a_ptr + 2 * stride_a);
  a5 = vld1q_f32(a_ptr + 2 * stride_a + 4);
  a6 = vld1q_f32(a_ptr + 3 * stride_a);
  a7 = vld1q_f32(a_ptr + 3 * stride_a + 4);
  a8 = vld1q_f32(a_ptr + 4 * stride_a);
  a9 = vld1q_f32(a_ptr + 4 * stride_a + 4);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);
  b4 = vld1q_f32(b_ptr + 4 * stride_b);
  b5 = vld1q_f32(b_ptr + 5 * stride_b);
  b6 = vld1q_f32(b_ptr + 6 * stride_b);
  b7 = vld1q_f32(b_ptr + 7 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);
  c2 = vld1q_f32(c_ptr + 2 * stride_c);
  c3 = vld1q_f32(c_ptr + 3 * stride_c);
  c4 = vld1q_f32(c_ptr + 4 * stride_c);

  MACE_GEMM_PART_CAL_8(0, 0, 1);
  MACE_GEMM_PART_CAL_8(1, 2, 3);
  MACE_GEMM_PART_CAL_8(2, 4, 5);
  MACE_GEMM_PART_CAL_8(3, 6, 7);
  MACE_GEMM_PART_CAL_8(4, 8, 9);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
  vst1q_f32(c_ptr + 2 * stride_c, c2);
  vst1q_f32(c_ptr + 3 * stride_c, c3);
  vst1q_f32(c_ptr + 4 * stride_c, c4);
#else
  GemmBlock(a_ptr, b_ptr, 5, 8, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm684(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11;
  float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
  float32x4_t c0, c1, c2, c3, c4, c5;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 4);
  a2 = vld1q_f32(a_ptr + 1 * stride_a);
  a3 = vld1q_f32(a_ptr + 1 * stride_a + 4);
  a4 = vld1q_f32(a_ptr + 2 * stride_a);
  a5 = vld1q_f32(a_ptr + 2 * stride_a + 4);
  a6 = vld1q_f32(a_ptr + 3 * stride_a);
  a7 = vld1q_f32(a_ptr + 3 * stride_a + 4);
  a8 = vld1q_f32(a_ptr + 4 * stride_a);
  a9 = vld1q_f32(a_ptr + 4 * stride_a + 4);
  a10 = vld1q_f32(a_ptr + 5 * stride_a);
  a11 = vld1q_f32(a_ptr + 5 * stride_a + 4);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);
  b4 = vld1q_f32(b_ptr + 4 * stride_b);
  b5 = vld1q_f32(b_ptr + 5 * stride_b);
  b6 = vld1q_f32(b_ptr + 6 * stride_b);
  b7 = vld1q_f32(b_ptr + 7 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);
  c2 = vld1q_f32(c_ptr + 2 * stride_c);
  c3 = vld1q_f32(c_ptr + 3 * stride_c);
  c4 = vld1q_f32(c_ptr + 4 * stride_c);
  c5 = vld1q_f32(c_ptr + 5 * stride_c);

  MACE_GEMM_PART_CAL_8(0, 0, 1);
  MACE_GEMM_PART_CAL_8(1, 2, 3);
  MACE_GEMM_PART_CAL_8(2, 4, 5);
  MACE_GEMM_PART_CAL_8(3, 6, 7);
  MACE_GEMM_PART_CAL_8(4, 8, 9);
  MACE_GEMM_PART_CAL_8(5, 10, 11);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
  vst1q_f32(c_ptr + 2 * stride_c, c2);
  vst1q_f32(c_ptr + 3 * stride_c, c3);
  vst1q_f32(c_ptr + 4 * stride_c, c4);
  vst1q_f32(c_ptr + 5 * stride_c, c5);

#else
  GemmBlock(a_ptr, b_ptr, 6, 8, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void Gemm784(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON)
  float32x4_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13;
  float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
  float32x4_t c0, c1, c2, c3, c4, c5, c6;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 4);
  a2 = vld1q_f32(a_ptr + 1 * stride_a);
  a3 = vld1q_f32(a_ptr + 1 * stride_a + 4);
  a4 = vld1q_f32(a_ptr + 2 * stride_a);
  a5 = vld1q_f32(a_ptr + 2 * stride_a + 4);
  a6 = vld1q_f32(a_ptr + 3 * stride_a);
  a7 = vld1q_f32(a_ptr + 3 * stride_a + 4);
  a8 = vld1q_f32(a_ptr + 4 * stride_a);
  a9 = vld1q_f32(a_ptr + 4 * stride_a + 4);
  a10 = vld1q_f32(a_ptr + 5 * stride_a);
  a11 = vld1q_f32(a_ptr + 5 * stride_a + 4);
  a12 = vld1q_f32(a_ptr + 6 * stride_a);
  a13 = vld1q_f32(a_ptr + 6 * stride_a + 4);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_b);
  b2 = vld1q_f32(b_ptr + 2 * stride_b);
  b3 = vld1q_f32(b_ptr + 3 * stride_b);
  b4 = vld1q_f32(b_ptr + 4 * stride_b);
  b5 = vld1q_f32(b_ptr + 5 * stride_b);
  b6 = vld1q_f32(b_ptr + 6 * stride_b);
  b7 = vld1q_f32(b_ptr + 7 * stride_b);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_c);
  c2 = vld1q_f32(c_ptr + 2 * stride_c);
  c3 = vld1q_f32(c_ptr + 3 * stride_c);
  c4 = vld1q_f32(c_ptr + 4 * stride_c);
  c5 = vld1q_f32(c_ptr + 5 * stride_c);
  c6 = vld1q_f32(c_ptr + 6 * stride_c);

  MACE_GEMM_PART_CAL_8(0, 0, 1);
  MACE_GEMM_PART_CAL_8(1, 2, 3);
  MACE_GEMM_PART_CAL_8(2, 4, 5);
  MACE_GEMM_PART_CAL_8(3, 6, 7);
  MACE_GEMM_PART_CAL_8(4, 8, 9);
  MACE_GEMM_PART_CAL_8(5, 10, 11);
  MACE_GEMM_PART_CAL_8(6, 12, 13);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_c, c1);
  vst1q_f32(c_ptr + 2 * stride_c, c2);
  vst1q_f32(c_ptr + 3 * stride_c, c3);
  vst1q_f32(c_ptr + 4 * stride_c, c4);
  vst1q_f32(c_ptr + 5 * stride_c, c5);
  vst1q_f32(c_ptr + 6 * stride_c, c6);

#else
  GemmBlock(a_ptr, b_ptr, 7, 8, 4, stride_a, stride_b, stride_c, c_ptr);
#endif
}

inline void GemmX84(const float *a_ptr,
                    const float *b_ptr,
                    const index_t stride_a,
                    const index_t stride_b,
                    const index_t stride_c,
                    float *c_ptr,
                    int row) {
  switch (row) {
    case 1:
      Gemm184(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 2:
      Gemm284(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 3:
      Gemm384(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 4:
      Gemm484(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 5:
      Gemm584(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 6:
      Gemm684(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 7:
      Gemm784(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    case 8:
      Gemm884(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      break;
    default:
      MACE_NOT_IMPLEMENTED;
  }
}

inline void GemmTile(const float *A,
                     const float *B,
                     const index_t height,
                     const index_t K,
                     const index_t width,
                     const index_t stride_a,
                     const index_t stride_b,
                     const index_t stride_c,
                     float *C) {
#if defined(MACE_ENABLE_NEON)
  index_t h = 0;
  index_t w = 0;
  index_t k = 0;
#if defined(__aarch64__)
  int reg_height_tile = 8;
  int reg_K_tile = 8;
#else
  int reg_height_tile = 6;
  int reg_K_tile = 4;
#endif

  for (h = 0; h < height - reg_height_tile + 1; h += reg_height_tile) {
    for (k = 0; k < K - reg_K_tile + 1; k += reg_K_tile) {
      const float *a_ptr = A + (h * stride_a + k);
#if defined(__aarch64__) && defined(__clang__)
      int nw = width >> 2;
      if (nw > 0) {
        // load A
        float32x4_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13,
            a14, a15;
        a0 = vld1q_f32(a_ptr);
        a1 = vld1q_f32(a_ptr + 4);
        a2 = vld1q_f32(a_ptr + 1 * stride_a);
        a3 = vld1q_f32(a_ptr + 1 * stride_a + 4);
        a4 = vld1q_f32(a_ptr + 2 * stride_a);
        a5 = vld1q_f32(a_ptr + 2 * stride_a + 4);
        a6 = vld1q_f32(a_ptr + 3 * stride_a);
        a7 = vld1q_f32(a_ptr + 3 * stride_a + 4);
        a8 = vld1q_f32(a_ptr + 4 * stride_a);
        a9 = vld1q_f32(a_ptr + 4 * stride_a + 4);
        a10 = vld1q_f32(a_ptr + 5 * stride_a);
        a11 = vld1q_f32(a_ptr + 5 * stride_a + 4);
        a12 = vld1q_f32(a_ptr + 6 * stride_a);
        a13 = vld1q_f32(a_ptr + 6 * stride_a + 4);
        a14 = vld1q_f32(a_ptr + 7 * stride_a);
        a15 = vld1q_f32(a_ptr + 7 * stride_a + 4);

        const float *b_ptr0 = B + k * stride_b;
        const float *b_ptr1 = B + (k + 1) * stride_b;
        const float *b_ptr2 = B + (k + 2) * stride_b;
        const float *b_ptr3 = B + (k + 3) * stride_b;
        const float *b_ptr4 = B + (k + 4) * stride_b;
        const float *b_ptr5 = B + (k + 5) * stride_b;
        const float *b_ptr6 = B + (k + 6) * stride_b;
        const float *b_ptr7 = B + (k + 7) * stride_b;

        float *c_ptr0 = C + h * stride_c;
        float *c_ptr1 = C + (h + 1) * stride_c;
        float *c_ptr2 = C + (h + 2) * stride_c;
        float *c_ptr3 = C + (h + 3) * stride_c;
        float *c_ptr4 = C + (h + 4) * stride_c;
        float *c_ptr5 = C + (h + 5) * stride_c;
        float *c_ptr6 = C + (h + 6) * stride_c;
        float *c_ptr7 = C + (h + 7) * stride_c;

        asm volatile(
            "prfm   pldl1keep, [%9, #128]       \n"
            "ld1    {v16.4s}, [%9], #16         \n"

            "prfm   pldl1keep, [%1, #128]       \n"
            "ld1    {v18.4s}, [%1]              \n"

            "prfm   pldl1keep, [%2, #128]       \n"
            "ld1    {v19.4s}, [%2]              \n"

            "0:                                 \n"

            "prfm   pldl1keep, [%3, #128]       \n"
            "ld1    {v20.4s}, [%3]              \n"
            "prfm   pldl1keep, [%4, #128]       \n"
            "ld1    {v21.4s}, [%4]              \n"
            "prfm   pldl1keep, [%5, #128]       \n"
            "ld1    {v22.4s}, [%5]              \n"
            "prfm   pldl1keep, [%6, #128]       \n"
            "ld1    {v23.4s}, [%6]              \n"
            "prfm   pldl1keep, [%7, #128]       \n"
            "ld1    {v24.4s}, [%7]              \n"
            "prfm   pldl1keep, [%8, #128]       \n"
            "ld1    {v25.4s}, [%8]              \n"
            "prfm   pldl1keep, [%10, #128]      \n"
            "ld1    {v17.4s}, [%10], #16        \n"

            "fmla   v18.4s, v16.4s, %34.s[0]    \n"
            "fmla   v19.4s, v16.4s, %35.s[0]    \n"
            "fmla   v20.4s, v16.4s, %36.s[0]    \n"
            "fmla   v21.4s, v16.4s, %37.s[0]    \n"

            "fmla   v22.4s, v16.4s, %38.s[0]    \n"
            "fmla   v23.4s, v16.4s, %39.s[0]    \n"
            "fmla   v24.4s, v16.4s, %40.s[0]    \n"
            "fmla   v25.4s, v16.4s, %41.s[0]    \n"

            "fmla   v18.4s, v17.4s, %34.s[1]    \n"
            "fmla   v19.4s, v17.4s, %35.s[1]    \n"
            "fmla   v20.4s, v17.4s, %36.s[1]    \n"
            "fmla   v21.4s, v17.4s, %37.s[1]    \n"

            "prfm   pldl1keep, [%11, #128]      \n"
            "ld1    {v16.4s}, [%11], #16        \n"

            "fmla   v22.4s, v17.4s, %38.s[1]    \n"
            "fmla   v23.4s, v17.4s, %39.s[1]    \n"
            "fmla   v24.4s, v17.4s, %40.s[1]    \n"
            "fmla   v25.4s, v17.4s, %41.s[1]    \n"

            "fmla   v18.4s, v16.4s, %34.s[2]    \n"
            "fmla   v19.4s, v16.4s, %35.s[2]    \n"
            "fmla   v20.4s, v16.4s, %36.s[2]    \n"
            "fmla   v21.4s, v16.4s, %37.s[2]    \n"

            "prfm   pldl1keep, [%12, #128]      \n"
            "ld1    {v17.4s}, [%12], #16        \n"

            "fmla   v22.4s, v16.4s, %38.s[2]    \n"
            "fmla   v23.4s, v16.4s, %39.s[2]    \n"
            "fmla   v24.4s, v16.4s, %40.s[2]    \n"
            "fmla   v25.4s, v16.4s, %41.s[2]    \n"

            "fmla   v18.4s, v17.4s, %34.s[3]    \n"
            "fmla   v19.4s, v17.4s, %35.s[3]    \n"
            "fmla   v20.4s, v17.4s, %36.s[3]    \n"
            "fmla   v21.4s, v17.4s, %37.s[3]    \n"

            "prfm   pldl1keep, [%13, #128]      \n"
            "ld1    {v16.4s}, [%13], #16        \n"

            "fmla   v22.4s, v17.4s, %38.s[3]    \n"
            "fmla   v23.4s, v17.4s, %39.s[3]    \n"
            "fmla   v24.4s, v17.4s, %40.s[3]    \n"
            "fmla   v25.4s, v17.4s, %41.s[3]    \n"

            "fmla   v18.4s, v16.4s, %42.s[0]    \n"
            "fmla   v19.4s, v16.4s, %43.s[0]    \n"
            "fmla   v20.4s, v16.4s, %44.s[0]    \n"
            "fmla   v21.4s, v16.4s, %45.s[0]    \n"

            "prfm   pldl1keep, [%14, #128]      \n"
            "ld1    {v17.4s}, [%14], #16        \n"

            "fmla   v22.4s, v16.4s, %46.s[0]    \n"
            "fmla   v23.4s, v16.4s, %47.s[0]    \n"
            "fmla   v24.4s, v16.4s, %48.s[0]    \n"
            "fmla   v25.4s, v16.4s, %49.s[0]    \n"

            "fmla   v18.4s, v17.4s, %42.s[1]    \n"
            "fmla   v19.4s, v17.4s, %43.s[1]    \n"
            "fmla   v20.4s, v17.4s, %44.s[1]    \n"
            "fmla   v21.4s, v17.4s, %45.s[1]    \n"

            "prfm   pldl1keep, [%15, #128]      \n"
            "ld1    {v16.4s}, [%15], #16        \n"

            "fmla   v22.4s, v17.4s, %46.s[1]    \n"
            "fmla   v23.4s, v17.4s, %47.s[1]    \n"
            "fmla   v24.4s, v17.4s, %48.s[1]    \n"
            "fmla   v25.4s, v17.4s, %49.s[1]    \n"

            "fmla   v18.4s, v16.4s, %42.s[2]    \n"
            "fmla   v19.4s, v16.4s, %43.s[2]    \n"
            "fmla   v20.4s, v16.4s, %44.s[2]    \n"
            "fmla   v21.4s, v16.4s, %45.s[2]    \n"

            "prfm   pldl1keep, [%16, #128]      \n"
            "ld1    {v17.4s}, [%16], #16        \n"

            "fmla   v22.4s, v16.4s, %46.s[2]    \n"
            "fmla   v23.4s, v16.4s, %47.s[2]    \n"
            "fmla   v24.4s, v16.4s, %48.s[2]    \n"
            "fmla   v25.4s, v16.4s, %49.s[2]    \n"

            "fmla   v18.4s, v17.4s, %42.s[3]    \n"
            "fmla   v19.4s, v17.4s, %43.s[3]    \n"
            "fmla   v20.4s, v17.4s, %44.s[3]    \n"
            "fmla   v21.4s, v17.4s, %45.s[3]    \n"

            "st1    {v18.4s}, [%1], #16         \n"
            "st1    {v19.4s}, [%2], #16         \n"
            "st1    {v20.4s}, [%3], #16         \n"
            "st1    {v21.4s}, [%4], #16         \n"

            "fmla   v22.4s, v17.4s, %46.s[3]    \n"
            "fmla   v23.4s, v17.4s, %47.s[3]    \n"
            "fmla   v24.4s, v17.4s, %48.s[3]    \n"
            "fmla   v25.4s, v17.4s, %49.s[3]    \n"

            "st1    {v22.4s}, [%5], #16         \n"
            "st1    {v23.4s}, [%6], #16         \n"
            "st1    {v24.4s}, [%7], #16         \n"
            "st1    {v25.4s}, [%8], #16         \n"

            "prfm   pldl1keep, [%9, #128]       \n"
            "ld1    {v16.4s}, [%9], #16         \n"
            "prfm   pldl1keep, [%1, #128]       \n"
            "ld1    {v18.4s}, [%1]              \n"
            "prfm   pldl1keep, [%2, #128]       \n"
            "ld1    {v19.4s}, [%2]              \n"

            "subs   %w0, %w0, #1                \n"
            "bne    0b                          \n"
            : "=r"(nw),      // 0
              "=r"(c_ptr0),  // 1
              "=r"(c_ptr1),  // 2
              "=r"(c_ptr2),  // 3
              "=r"(c_ptr3),  // 4
              "=r"(c_ptr4),  // 5
              "=r"(c_ptr5),  // 6
              "=r"(c_ptr6),  // 7
              "=r"(c_ptr7),  // 8
              "=r"(b_ptr0),  // 9
              "=r"(b_ptr1),  // 10
              "=r"(b_ptr2),  // 11
              "=r"(b_ptr3),  // 12
              "=r"(b_ptr4),  // 13
              "=r"(b_ptr5),  // 14
              "=r"(b_ptr6),  // 15
              "=r"(b_ptr7)   // 16
            : "0"(nw),       // 17
              "1"(c_ptr0),   // 18
              "2"(c_ptr1),   // 19
              "3"(c_ptr2),   // 20
              "4"(c_ptr3),   // 21
              "5"(c_ptr4),   // 22
              "6"(c_ptr5),   // 23
              "7"(c_ptr6),   // 24
              "8"(c_ptr7),   // 25
              "9"(b_ptr0),   // 26
              "10"(b_ptr1),  // 27
              "11"(b_ptr2),  // 28
              "12"(b_ptr3),  // 29
              "13"(b_ptr4),  // 30
              "14"(b_ptr5),  // 31
              "15"(b_ptr6),  // 32
              "16"(b_ptr7),  // 33
              "w"(a0),       // 34
              "w"(a2),       // 35
              "w"(a4),       // 36
              "w"(a6),       // 37
              "w"(a8),       // 38
              "w"(a10),      // 39
              "w"(a12),      // 40
              "w"(a14),      // 41
              "w"(a1),       // 42
              "w"(a3),       // 43
              "w"(a5),       // 44
              "w"(a7),       // 45
              "w"(a9),       // 46
              "w"(a11),      // 47
              "w"(a13),      // 48
              "w"(a15)       // 49
            : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
              "v23", "v24", "v25");

        w = (width >> 2) << 2;
      }
#elif defined(__aarch64__)  // gcc
      for (w = 0; w + 3 < width; w += 4) {
        const float *b_ptr = B + (k * stride_b + w);
        float *c_ptr = C + (h * stride_c + w);
        Gemm884(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr);
      }
#else                       // armv7
      int nw = width >> 2;
      if (nw > 0) {
        float32x4_t a0, a1, a2, a3, a4, a5;

        a0 = vld1q_f32(a_ptr);
        a1 = vld1q_f32(a_ptr + 1 * stride_a);
        a2 = vld1q_f32(a_ptr + 2 * stride_a);
        a3 = vld1q_f32(a_ptr + 3 * stride_a);
        a4 = vld1q_f32(a_ptr + 4 * stride_a);
        a5 = vld1q_f32(a_ptr + 5 * stride_a);

        const float *b_ptr0 = B + k * stride_b;
        const float *b_ptr1 = B + (k + 1) * stride_b;
        const float *b_ptr2 = B + (k + 2) * stride_b;
        const float *b_ptr3 = B + (k + 3) * stride_b;

        float *c_ptr0 = C + h * stride_c;
        float *c_ptr1 = C + (h + 1) * stride_c;
        float *c_ptr2 = C + (h + 2) * stride_c;
        float *c_ptr3 = C + (h + 3) * stride_c;
        float *c_ptr4 = C + (h + 4) * stride_c;
        float *c_ptr5 = C + (h + 5) * stride_c;

        asm volatile(
            "pld    [%7, #128]                  \n"
            "vld1.f32    {d12-d13}, [%7]!       \n"
            "pld    [%1, #128]                  \n"
            "vld1.f32    {d16-d17}, [%1]        \n"
            "pld    [%2, #128]                  \n"
            "vld1.f32    {d18-d19}, [%2]        \n"

            "0:                                 \n"

            "pld    [%3, #128]                  \n"
            "vld1.f32    {d20-d21}, [%3]        \n"
            "pld    [%4, #128]                  \n"
            "vld1.f32    {d22-d23}, [%4]        \n"
            "pld    [%5, #128]                  \n"
            "vld1.f32    {d24-d25}, [%5]        \n"
            "pld    [%6, #128]                  \n"
            "vld1.f32    {d26-d27}, [%6]        \n"

            "pld    [%8, #128]                  \n"
            "vld1.f32    {d14-d15}, [%8]!       \n"

            "vmla.f32   q8, q6, %e22[0]         \n"
            "vmla.f32   q9, q6, %e23[0]         \n"
            "vmla.f32   q10, q6, %e24[0]        \n"
            "vmla.f32   q11, q6, %e25[0]        \n"
            "vmla.f32   q12, q6, %e26[0]        \n"
            "vmla.f32   q13, q6, %e27[0]        \n"

            "pld    [%9, #128]                  \n"
            "vld1.f32    {d12-d13}, [%9]!       \n"

            "vmla.f32   q8, q7, %e22[1]         \n"
            "vmla.f32   q9, q7, %e23[1]         \n"
            "vmla.f32   q10, q7, %e24[1]        \n"
            "vmla.f32   q11, q7, %e25[1]        \n"
            "vmla.f32   q12, q7, %e26[1]        \n"
            "vmla.f32   q13, q7, %e27[1]        \n"

            "pld    [%10, #128]                 \n"
            "vld1.f32    {d14-d15}, [%10]!      \n"

            "vmla.f32   q8, q6, %f22[0]         \n"
            "vmla.f32   q9, q6, %f23[0]         \n"
            "vmla.f32   q10, q6, %f24[0]        \n"
            "vmla.f32   q11, q6, %f25[0]        \n"
            "vmla.f32   q12, q6, %f26[0]        \n"
            "vmla.f32   q13, q6, %f27[0]        \n"

            "vmla.f32   q8, q7, %f22[1]         \n"
            "vmla.f32   q9, q7, %f23[1]         \n"
            "vmla.f32   q10, q7, %f24[1]        \n"
            "vmla.f32   q11, q7, %f25[1]        \n"
            "vmla.f32   q12, q7, %f26[1]        \n"
            "vmla.f32   q13, q7, %f27[1]        \n"

            "vst1.f32   {d16-d17}, [%1]!        \n"
            "vst1.f32   {d18-d19}, [%2]!        \n"

            "pld    [%7, #128]                  \n"
            "vld1.f32    {d12-d13}, [%7]!       \n"

            "vst1.f32   {d20-d21}, [%3]!        \n"
            "vst1.f32   {d22-d23}, [%4]!        \n"

            "pld    [%1, #128]                  \n"
            "vld1.f32    {d16-d17}, [%1]        \n"

            "vst1.f32   {d24-d25}, [%5]!        \n"
            "vst1.f32   {d26-d27}, [%6]!        \n"

            "pld    [%2, #128]                  \n"
            "vld1.f32    {d18-d19}, [%2]        \n"

            "subs   %0, #1                      \n"
            "bne    0b                          \n"
            : "=r"(nw),      // 0
              "=r"(c_ptr0),  // 1
              "=r"(c_ptr1),  // 2
              "=r"(c_ptr2),  // 3
              "=r"(c_ptr3),  // 4
              "=r"(c_ptr4),  // 5
              "=r"(c_ptr5),  // 6
              "=r"(b_ptr0),  // 7
              "=r"(b_ptr1),  // 8
              "=r"(b_ptr2),  // 9
              "=r"(b_ptr3)   // 10
            : "0"(nw),       // 11
              "1"(c_ptr0),   // 12
              "2"(c_ptr1),   // 13
              "3"(c_ptr2),   // 14
              "4"(c_ptr3),   // 15
              "5"(c_ptr4),   // 16
              "6"(c_ptr5),   // 17
              "7"(b_ptr0),   // 18
              "8"(b_ptr1),   // 19
              "9"(b_ptr2),   // 20
              "10"(b_ptr3),  // 21
              "w"(a0),       // 22
              "w"(a1),       // 23
              "w"(a2),       // 24
              "w"(a3),       // 25
              "w"(a4),       // 26
              "w"(a5)        // 27
            : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12",
              "q13", "q14", "q15");

        w = (width >> 2) << 2;
      }
#endif
      if (w < width) {
        const float *b_ptr = B + (k * stride_b + w);
        float *c_ptr = C + (h * stride_c + w);
        GemmBlock(a_ptr, b_ptr, reg_height_tile, reg_K_tile, width - w,
                  stride_a, stride_b, stride_c, c_ptr);
      }
    }
    if (k < K) {
      const float *a_ptr = A + (h * stride_a + k);
      const float *b_ptr = B + k * stride_b;
      float *c_ptr = C + h * stride_c;
      GemmBlock(a_ptr, b_ptr, reg_height_tile, K - k, width, stride_a, stride_b,
                stride_c, c_ptr);
    }
  }
  if (h < height) {
    index_t remain_h = height - h;
    for (k = 0; k < K - reg_K_tile; k += reg_K_tile) {
      const float *a_ptr = A + (h * stride_a + k);
      index_t w;
      for (w = 0; w + 3 < width; w += 4) {
        const float *b_ptr = B + (k * stride_b + w);
        float *c_ptr = C + (h * stride_c + w);
#if defined(__aarch64__)
        GemmX84(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr, remain_h);
#else
        GemmX44(a_ptr, b_ptr, stride_a, stride_b, stride_c, c_ptr, remain_h);
#endif
      }
      if (w < width) {
        const float *b_ptr = B + (k * stride_b + w);
        float *c_ptr = C + (h * stride_c + w);
        GemmBlock(a_ptr, b_ptr, remain_h, reg_K_tile, width - w, stride_a,
                  stride_b, stride_c, c_ptr);
      }
    }
    if (k < K) {
      const float *a_ptr = A + (h * stride_a + k);
      const float *b_ptr = B + k * stride_b;
      float *c_ptr = C + h * stride_c;
      GemmBlock(a_ptr, b_ptr, remain_h, K - k, width, stride_a, stride_b,
                stride_c, c_ptr);
    }
  }
#else
  GemmBlock(A, B, height, K, width, stride_a, stride_b, stride_c, C);
#endif  // MACE_ENABLE_NEON
}

void Transpose(const float *src,
               index_t height,
               index_t width,
               index_t stride_w,
               float *dst) {
  for (index_t h = 0; h < height; ++h) {
    for (index_t w = 0; w < width; ++w) {
      dst[w * height + h] = src[h * stride_w + w];
    }
  }
}

}  // namespace

// A: height x K, B: K x width, C: height x width
void Gemm(const float *A,
          const float *B,
          const index_t batch,
          const index_t height,
          const index_t K,
          const index_t width,
          float *C,
          const bool transpose_a,
          const bool transpose_b) {
  if (width == 1) {
    for (index_t b = 0; b < batch; ++b) {
      Gemv(A + b * height * K, B + b * K, 1, K, height, C + b * height);
    }
    return;
  }
  memset(C, 0, sizeof(float) * batch * height * width);

  // It is better to use large block size if it fits for fast cache.
  // Assume l1 cache size is 32k, we load three blocks at a time (A, B, C),
  // the block size should be sqrt(32k / sizeof(T) / 3).
  // As number of input channels of convolution is normally power of 2, and
  // we have not optimized tiling remains, we use the following magic number
  const index_t block_size = 64;
  const index_t block_tile_height = RoundUpDiv(height, block_size);
  const index_t block_tile_width = RoundUpDiv(width, block_size);
  const index_t block_tile_k = RoundUpDiv(K, block_size);
  const index_t block_tile[3] = {block_tile_height, block_tile_width,
                                 block_tile_k};
  const index_t remain_height = height % block_size;
  const index_t remain_width = width % block_size;
  const index_t remain_k = K % block_size;
  const index_t remain[3] = {remain_height, remain_width, remain_k};

#pragma omp parallel for collapse(3)
  for (index_t n = 0; n < batch; ++n) {
    for (index_t bh = 0; bh < block_tile[0]; ++bh) {
      for (index_t bw = 0; bw < block_tile[1]; ++bw) {
        const float *a_base = A + n * height * K;
        const float *b_base = B + n * K * width;
        float *c_base = C + n * height * width;

        const index_t ih_begin = bh * block_size;
        const index_t ih_end =
            bh * block_size +
            (bh == block_tile[0] - 1 && remain[0] > 0 ? remain[0] : block_size);
        const index_t iw_begin = bw * block_size;
        const index_t iw_end =
            bw * block_size +
            (bw == block_tile[1] - 1 && remain[1] > 0 ? remain[1] : block_size);

        for (index_t bk = 0; bk < block_tile[2]; ++bk) {
          const index_t ik_begin = bk * block_size;
          const index_t ik_end =
              bk * block_size + (bk == block_tile[2] - 1 && remain[2] > 0
                                     ? remain[2]
                                     : block_size);

          Tensor trans_a;
          Tensor trans_b;
          const float *real_a = nullptr;
          const float *real_b = nullptr;
          float *real_c = c_base + (ih_begin * width + iw_begin);
          index_t stride_a;
          index_t stride_b;
          index_t stride_c = width;

          if (transpose_a) {
            trans_a.Resize({block_size, block_size});
            float *trans_a_data = trans_a.mutable_data<float>();
            // A[K, H] -> A[H, K]
            Transpose(a_base + (ik_begin * height + ih_begin),
                      ik_end - ik_begin, ih_end - ih_begin, height,
                      trans_a_data);
            real_a = trans_a_data;
            stride_a = ik_end - ik_begin;
          } else {
            real_a = a_base + (ih_begin * K + ik_begin);
            stride_a = K;
          }

          if (transpose_b) {
            trans_b.Resize({block_size, block_size});
            float *trans_b_data = trans_b.mutable_data<float>();
            // B[W, K] -> B[K, W]
            Transpose(b_base + (iw_begin * K + ik_begin), iw_end - iw_begin,
                      ik_end - ik_begin, K, trans_b_data);
            real_b = trans_b_data;
            stride_b = iw_end - iw_begin;
          } else {
            real_b = b_base + (ik_begin * width + iw_begin);
            stride_b = width;
          }

          // inside block:
          // calculate C[bh, bw] += A[bh, bk] * B[bk, bw] for one k
          GemmTile(real_a, real_b, ih_end - ih_begin, ik_end - ik_begin,
                   iw_end - iw_begin, stride_a, stride_b, stride_c, real_c);
        }  // bk
      }    // bw
    }      // bh
  }        // n
}

// A: height x K, B: K x width, C: height x width
void GemmRef(const float *A,
             const float *B,
             const index_t batch,
             const index_t height,
             const index_t K,
             const index_t width,
             float *C,
             const bool transpose_a,
             const bool transpose_b) {
  memset(C, 0, sizeof(float) * batch * height * width);

  Tensor trans_a;
  Tensor trans_b;
  float *trans_a_data = nullptr;
  float *trans_b_data = nullptr;
  if (transpose_a) {
    trans_a.Resize({height, K});
    trans_a_data = trans_a.mutable_data<float>();
  }
  if (transpose_b) {
    trans_b.Resize({K, width});
    trans_b_data = trans_b.mutable_data<float>();
  }

  for (index_t b = 0; b < batch; ++b) {
    const float *real_a = nullptr;
    const float *real_b = nullptr;
    float *real_c = C + b * height * width;
    if (transpose_a) {
      // A[K, H] -> A[H, K]
      Transpose(A + b * height * K, K, height, height, trans_a_data);
      real_a = trans_a_data;
    } else {
      real_a = A + b * height * K;
    }
    if (transpose_b) {
      // B[W, K] -> B[K, W]
      Transpose(B + b * width * K, width, K, K, trans_b_data);
      real_b = trans_b_data;
    } else {
      real_b = B + b * width * K;
    }

    for (index_t i = 0; i < height; ++i) {
      for (index_t j = 0; j < width; ++j) {
        for (index_t k = 0; k < K; ++k) {
          real_c[i * width + j] += real_a[i * K + k] * real_b[k * width + j];
        }
      }
    }
  }
}

void GemvRef(const float *m_ptr,
             const float *v_ptr,
             const index_t batch,
             const index_t width,
             const index_t height,
             float *out_ptr) {
  memset(out_ptr, 0, batch * height * sizeof(float));
#pragma omp parallel for collapse(2)
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        out_ptr[b * height + h] += v_ptr[b * width + w] * m_ptr[h * width + w];
      }
    }
  }
}

// TODO(liyin): batched gemv can be transformed to gemm (w/ transpose)
void Gemv(const float *m_ptr,
          const float *v_ptr,
          const index_t batch,
          const index_t width,
          const index_t height,
          float *out_ptr) {
#if defined(MACE_ENABLE_NEON)
// TODO(liyin/wch): try height tiling = 8
#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < batch; ++b) {
    for (index_t h = 0; h < height; h += 4) {
      if (h + 3 < height) {
        const float *m_ptr0 = m_ptr + h * width;
        const float *m_ptr1 = m_ptr0 + width;
        const float *m_ptr2 = m_ptr1 + width;
        const float *m_ptr3 = m_ptr2 + width;
        const float *v_ptr0 = v_ptr + b * width;
        float *out_ptr0 = out_ptr + b * height + h;

        float32x4_t vm0, vm1, vm2, vm3;
        float32x4_t vv;

        float32x4_t vsum0 = vdupq_n_f32(0.f);
        float32x4_t vsum1 = vdupq_n_f32(0.f);
        float32x4_t vsum2 = vdupq_n_f32(0.f);
        float32x4_t vsum3 = vdupq_n_f32(0.f);

        index_t w;
        for (w = 0; w + 3 < width; w += 4) {
          vm0 = vld1q_f32(m_ptr0);
          vm1 = vld1q_f32(m_ptr1);
          vm2 = vld1q_f32(m_ptr2);
          vm3 = vld1q_f32(m_ptr3);
          vv = vld1q_f32(v_ptr0);

          vsum0 = vmlaq_f32(vsum0, vm0, vv);
          vsum1 = vmlaq_f32(vsum1, vm1, vv);
          vsum2 = vmlaq_f32(vsum2, vm2, vv);
          vsum3 = vmlaq_f32(vsum3, vm3, vv);

          m_ptr0 += 4;
          m_ptr1 += 4;
          m_ptr2 += 4;
          m_ptr3 += 4;
          v_ptr0 += 4;
        }
        float sum0 = vaddvq_f32(vsum0);
        float sum1 = vaddvq_f32(vsum1);
        float sum2 = vaddvq_f32(vsum2);
        float sum3 = vaddvq_f32(vsum3);

        // handle remaining w
        for (; w < width; ++w) {
          sum0 += m_ptr0[0] * v_ptr0[0];
          sum1 += m_ptr1[0] * v_ptr0[0];
          sum2 += m_ptr2[0] * v_ptr0[0];
          sum3 += m_ptr3[0] * v_ptr0[0];
          m_ptr0++;
          m_ptr1++;
          m_ptr2++;
          m_ptr3++;
          v_ptr0++;
        }
        *out_ptr0++ = sum0;
        *out_ptr0++ = sum1;
        *out_ptr0++ = sum2;
        *out_ptr0++ = sum3;
      } else {
        for (index_t hh = h; hh < height; ++hh) {
          float32x4_t vsum0 = vdupq_n_f32(0.f);
          const float *m_ptr0 = m_ptr + hh * width;
          const float *v_ptr0 = v_ptr + b * width;
          index_t w;
          for (w = 0; w + 3 < width; w += 4) {
            float32x4_t vm = vld1q_f32(m_ptr0);
            float32x4_t vv = vld1q_f32(v_ptr0);
            vsum0 = vmlaq_f32(vsum0, vm, vv);
            m_ptr0 += 4;
            v_ptr0 += 4;
          }
          float sum = vaddvq_f32(vsum0);
          for (; w < width; ++w) {
            sum += m_ptr0[0] * v_ptr0[0];
            m_ptr0++;
            v_ptr0++;
          }
          out_ptr[b * height + hh] = sum;
        }
      }  // if
    }    // h
  }      // b
#else
  GemvRef(m_ptr, v_ptr, batch, width, height, out_ptr);
#endif
}

}  // namespace kernels
}  // namespace mace
