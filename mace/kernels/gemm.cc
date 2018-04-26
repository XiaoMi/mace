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

#include <math.h>
#include <algorithm>

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#include "mace/kernels/gemm.h"
#include "mace/utils/utils.h"
#include "mace/utils/logging.h"



namespace mace {
namespace kernels {

namespace {
inline void GemmBlock(const float *A,
                      const float *B,
                      const index_t height,
                      const index_t K,
                      const index_t width,
                      const index_t stride_k,
                      const index_t stride_w,
                      float *C) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * stride_w + j] += A[i * stride_k + k] * B[k * stride_w + j];
      }
    }
  }
}

// TODO(liyin): may need implement 883 since RGB
inline void Gemm884(const float *a_ptr,
                    const float *b_ptr,
                    index_t stride_k,
                    index_t stride_w,
                    float *c_ptr) {
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
  float32x4_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
    a15;
  float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
  float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;

  a0 = vld1q_f32(a_ptr);
  a1 = vld1q_f32(a_ptr + 4);
  a2 = vld1q_f32(a_ptr + 1 * stride_k);
  a3 = vld1q_f32(a_ptr + 1 * stride_k + 4);
  a4 = vld1q_f32(a_ptr + 2 * stride_k);
  a5 = vld1q_f32(a_ptr + 2 * stride_k + 4);
  a6 = vld1q_f32(a_ptr + 3 * stride_k);
  a7 = vld1q_f32(a_ptr + 3 * stride_k + 4);
  a8 = vld1q_f32(a_ptr + 4 * stride_k);
  a9 = vld1q_f32(a_ptr + 4 * stride_k + 4);
  a10 = vld1q_f32(a_ptr + 5 * stride_k);
  a11 = vld1q_f32(a_ptr + 5 * stride_k + 4);
  a12 = vld1q_f32(a_ptr + 6 * stride_k);
  a13 = vld1q_f32(a_ptr + 6 * stride_k + 4);
  a14 = vld1q_f32(a_ptr + 7 * stride_k);
  a15 = vld1q_f32(a_ptr + 7 * stride_k + 4);

  b0 = vld1q_f32(b_ptr);
  b1 = vld1q_f32(b_ptr + 1 * stride_w);
  b2 = vld1q_f32(b_ptr + 2 * stride_w);
  b3 = vld1q_f32(b_ptr + 3 * stride_w);
  b4 = vld1q_f32(b_ptr + 4 * stride_w);
  b5 = vld1q_f32(b_ptr + 5 * stride_w);
  b6 = vld1q_f32(b_ptr + 6 * stride_w);
  b7 = vld1q_f32(b_ptr + 7 * stride_w);

  c0 = vld1q_f32(c_ptr);
  c1 = vld1q_f32(c_ptr + 1 * stride_w);
  c2 = vld1q_f32(c_ptr + 2 * stride_w);
  c3 = vld1q_f32(c_ptr + 3 * stride_w);
  c4 = vld1q_f32(c_ptr + 4 * stride_w);
  c5 = vld1q_f32(c_ptr + 5 * stride_w);
  c6 = vld1q_f32(c_ptr + 6 * stride_w);
  c7 = vld1q_f32(c_ptr + 7 * stride_w);

#define MACE_CONV_1x1_REG_CAL(RC, RA, RAN) \
  c##RC = vfmaq_laneq_f32(c##RC, b0, a##RA, 0); \
  c##RC = vfmaq_laneq_f32(c##RC, b1, a##RA, 1); \
  c##RC = vfmaq_laneq_f32(c##RC, b2, a##RA, 2); \
  c##RC = vfmaq_laneq_f32(c##RC, b3, a##RA, 3); \
  c##RC = vfmaq_laneq_f32(c##RC, b4, a##RAN, 0); \
  c##RC = vfmaq_laneq_f32(c##RC, b5, a##RAN, 1); \
  c##RC = vfmaq_laneq_f32(c##RC, b6, a##RAN, 2); \
  c##RC = vfmaq_laneq_f32(c##RC, b7, a##RAN, 3);

  MACE_CONV_1x1_REG_CAL(0, 0, 1);
  MACE_CONV_1x1_REG_CAL(1, 2, 3);
  MACE_CONV_1x1_REG_CAL(2, 4, 5);
  MACE_CONV_1x1_REG_CAL(3, 6, 7);
  MACE_CONV_1x1_REG_CAL(4, 8, 9);
  MACE_CONV_1x1_REG_CAL(5, 10, 11);
  MACE_CONV_1x1_REG_CAL(6, 12, 13);
  MACE_CONV_1x1_REG_CAL(7, 14, 15);

  vst1q_f32(c_ptr, c0);
  vst1q_f32(c_ptr + 1 * stride_w, c1);
  vst1q_f32(c_ptr + 2 * stride_w, c2);
  vst1q_f32(c_ptr + 3 * stride_w, c3);
  vst1q_f32(c_ptr + 4 * stride_w, c4);
  vst1q_f32(c_ptr + 5 * stride_w, c5);
  vst1q_f32(c_ptr + 6 * stride_w, c6);
  vst1q_f32(c_ptr + 7 * stride_w, c7);

#else
  GemmBlock(a_ptr, b_ptr, 8, 8, 4, stride_k, stride_w, c_ptr);
#endif
}

inline void GemmTile(const float *A,
                     const float *B,
                     const index_t height,
                     const index_t K,
                     const index_t width,
                     const index_t stride_k,
                     const index_t stride_w,
                     float *C) {
  index_t h, w, k;

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
  for (h = 0; h + 7 < height; h += 8) {
    for (k = 0; k + 7 < K; k += 8) {
      const float *a_ptr = A + (h * stride_k + k);
#ifdef __clang__
      int nw = width >> 2;
      if (nw > 0) {
        // load A
        float32x4_t a0, a1, a2, a3, a4, a5, a6, a7,
          a8, a9, a10, a11, a12, a13, a14, a15;
        a0 = vld1q_f32(a_ptr);
        a1 = vld1q_f32(a_ptr + 4);
        a2 = vld1q_f32(a_ptr + 1 * stride_k);
        a3 = vld1q_f32(a_ptr + 1 * stride_k + 4);
        a4 = vld1q_f32(a_ptr + 2 * stride_k);
        a5 = vld1q_f32(a_ptr + 2 * stride_k + 4);
        a6 = vld1q_f32(a_ptr + 3 * stride_k);
        a7 = vld1q_f32(a_ptr + 3 * stride_k + 4);
        a8 = vld1q_f32(a_ptr + 4 * stride_k);
        a9 = vld1q_f32(a_ptr + 4 * stride_k + 4);
        a10 = vld1q_f32(a_ptr + 5 * stride_k);
        a11 = vld1q_f32(a_ptr + 5 * stride_k + 4);
        a12 = vld1q_f32(a_ptr + 6 * stride_k);
        a13 = vld1q_f32(a_ptr + 6 * stride_k + 4);
        a14 = vld1q_f32(a_ptr + 7 * stride_k);
        a15 = vld1q_f32(a_ptr + 7 * stride_k + 4);

        const float *b_ptr0 = B + k * stride_w;
        const float *b_ptr1 = B + (k + 1) * stride_w;
        const float *b_ptr2 = B + (k + 2) * stride_w;
        const float *b_ptr3 = B + (k + 3) * stride_w;
        const float *b_ptr4 = B + (k + 4) * stride_w;
        const float *b_ptr5 = B + (k + 5) * stride_w;
        const float *b_ptr6 = B + (k + 6) * stride_w;
        const float *b_ptr7 = B + (k + 7) * stride_w;

        float *c_ptr0 = C + h * stride_w;
        float *c_ptr1 = C + (h + 1) * stride_w;
        float *c_ptr2 = C + (h + 2) * stride_w;
        float *c_ptr3 = C + (h + 3) * stride_w;
        float *c_ptr4 = C + (h + 4) * stride_w;
        float *c_ptr5 = C + (h + 5) * stride_w;
        float *c_ptr6 = C + (h + 6) * stride_w;
        float *c_ptr7 = C + (h + 7) * stride_w;

        asm volatile(
        "0:                                 \n"

          "prfm   pldl1keep, [%1, #128]       \n"
          "ld1    {v24.4s}, [%1]              \n"

          // load b: 0-7
          "prfm   pldl1keep, [%9, #128]       \n"
          "ld1    {v16.4s}, [%9], #16         \n"

          "prfm   pldl1keep, [%10, #128]      \n"
          "ld1    {v17.4s}, [%10], #16        \n"

          "prfm   pldl1keep, [%11, #128]      \n"
          "ld1    {v18.4s}, [%11], #16        \n"

          "prfm   pldl1keep, [%12, #128]      \n"
          "ld1    {v19.4s}, [%12], #16        \n"

          "prfm   pldl1keep, [%2, #128]       \n"
          "ld1    {v25.4s}, [%2]              \n"

          "prfm   pldl1keep, [%13, #128]      \n"
          "ld1    {v20.4s}, [%13], #16        \n"

          "prfm   pldl1keep, [%14, #128]      \n"
          "ld1    {v21.4s}, [%14], #16        \n"

          "prfm   pldl1keep, [%15, #128]      \n"
          "ld1    {v22.4s}, [%15], #16        \n"

          "prfm   pldl1keep, [%16, #128]      \n"
          "ld1    {v23.4s}, [%16], #16        \n"

          "prfm   pldl1keep, [%3, #128]       \n"
          "ld1    {v26.4s}, [%3]              \n"

          "fmla   v24.4s, v16.4s, %34.s[0]    \n"
          "fmla   v24.4s, v17.4s, %34.s[1]    \n"
          "fmla   v24.4s, v18.4s, %34.s[2]    \n"
          "fmla   v24.4s, v19.4s, %34.s[3]    \n"

          "fmla   v24.4s, v20.4s, %35.s[0]    \n"
          "fmla   v24.4s, v21.4s, %35.s[1]    \n"
          "fmla   v24.4s, v22.4s, %35.s[2]    \n"
          "fmla   v24.4s, v23.4s, %35.s[3]    \n"

          "st1    {v24.4s}, [%1], #16         \n"

          "fmla   v25.4s, v16.4s, %36.s[0]    \n"
          "fmla   v25.4s, v17.4s, %36.s[1]    \n"
          "fmla   v25.4s, v18.4s, %36.s[2]    \n"
          "fmla   v25.4s, v19.4s, %36.s[3]    \n"

          "fmla   v25.4s, v20.4s, %37.s[0]    \n"
          "fmla   v25.4s, v21.4s, %37.s[1]    \n"
          "fmla   v25.4s, v22.4s, %37.s[2]    \n"
          "fmla   v25.4s, v23.4s, %37.s[3]    \n"

          "prfm   pldl1keep, [%4, #128]       \n"
          "ld1    {v24.4s}, [%4]              \n"

          "st1    {v25.4s}, [%2], #16         \n"

          "fmla   v26.4s, v16.4s, %38.s[0]    \n"
          "fmla   v26.4s, v17.4s, %38.s[1]    \n"
          "fmla   v26.4s, v18.4s, %38.s[2]    \n"
          "fmla   v26.4s, v19.4s, %38.s[3]    \n"

          "fmla   v26.4s, v20.4s, %39.s[0]    \n"
          "fmla   v26.4s, v21.4s, %39.s[1]    \n"
          "fmla   v26.4s, v22.4s, %39.s[2]    \n"
          "fmla   v26.4s, v23.4s, %39.s[3]    \n"

          "prfm   pldl1keep, [%5, #128]       \n"
          "ld1    {v25.4s}, [%5]              \n"

          "st1    {v26.4s}, [%3], #16         \n"

          "fmla   v24.4s, v16.4s, %40.s[0]    \n"
          "fmla   v24.4s, v17.4s, %40.s[1]    \n"
          "fmla   v24.4s, v18.4s, %40.s[2]    \n"
          "fmla   v24.4s, v19.4s, %40.s[3]    \n"

          "fmla   v24.4s, v20.4s, %41.s[0]    \n"
          "fmla   v24.4s, v21.4s, %41.s[1]    \n"
          "fmla   v24.4s, v22.4s, %41.s[2]    \n"
          "fmla   v24.4s, v23.4s, %41.s[3]    \n"

          "prfm   pldl1keep, [%6, #128]       \n"
          "ld1    {v26.4s}, [%6]              \n"

          "st1    {v24.4s}, [%4], #16         \n"

          "fmla   v25.4s, v16.4s, %42.s[0]    \n"
          "fmla   v25.4s, v17.4s, %42.s[1]    \n"
          "fmla   v25.4s, v18.4s, %42.s[2]    \n"
          "fmla   v25.4s, v19.4s, %42.s[3]    \n"

          "fmla   v25.4s, v20.4s, %43.s[0]    \n"
          "fmla   v25.4s, v21.4s, %43.s[1]    \n"
          "fmla   v25.4s, v22.4s, %43.s[2]    \n"
          "fmla   v25.4s, v23.4s, %43.s[3]    \n"

          "prfm   pldl1keep, [%7, #128]       \n"
          "ld1    {v24.4s}, [%7]              \n"

          "st1    {v25.4s}, [%5], #16         \n"

          "fmla   v26.4s, v16.4s, %44.s[0]    \n"
          "fmla   v26.4s, v17.4s, %44.s[1]    \n"
          "fmla   v26.4s, v18.4s, %44.s[2]    \n"
          "fmla   v26.4s, v19.4s, %44.s[3]    \n"

          "fmla   v26.4s, v20.4s, %45.s[0]    \n"
          "fmla   v26.4s, v21.4s, %45.s[1]    \n"
          "fmla   v26.4s, v22.4s, %45.s[2]    \n"
          "fmla   v26.4s, v23.4s, %45.s[3]    \n"

          "prfm   pldl1keep, [%8, #128]       \n"
          "ld1    {v25.4s}, [%8]              \n"

          "st1    {v26.4s}, [%6], #16         \n"

          "fmla   v24.4s, v16.4s, %46.s[0]    \n"
          "fmla   v24.4s, v17.4s, %46.s[1]    \n"
          "fmla   v24.4s, v18.4s, %46.s[2]    \n"
          "fmla   v24.4s, v19.4s, %46.s[3]    \n"

          "fmla   v24.4s, v20.4s, %47.s[0]    \n"
          "fmla   v24.4s, v21.4s, %47.s[1]    \n"
          "fmla   v24.4s, v22.4s, %47.s[2]    \n"
          "fmla   v24.4s, v23.4s, %47.s[3]    \n"

          "st1    {v24.4s}, [%7], #16         \n"

          "fmla   v25.4s, v16.4s, %48.s[0]    \n"
          "fmla   v25.4s, v17.4s, %48.s[1]    \n"
          "fmla   v25.4s, v18.4s, %48.s[2]    \n"
          "fmla   v25.4s, v19.4s, %48.s[3]    \n"

          "fmla   v25.4s, v20.4s, %49.s[0]    \n"
          "fmla   v25.4s, v21.4s, %49.s[1]    \n"
          "fmla   v25.4s, v22.4s, %49.s[2]    \n"
          "fmla   v25.4s, v23.4s, %49.s[3]    \n"

          "st1    {v25.4s}, [%8], #16         \n"

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
        "w"(a1),       // 35
        "w"(a2),       // 36
        "w"(a3),       // 37
        "w"(a4),       // 38
        "w"(a5),       // 39
        "w"(a6),       // 40
        "w"(a7),       // 41
        "w"(a8),       // 42
        "w"(a9),       // 43
        "w"(a10),      // 44
        "w"(a11),      // 45
        "w"(a12),      // 46
        "w"(a13),      // 47
        "w"(a14),      // 48
        "w"(a15)       // 49
        : "cc", "memory",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23",
        "v24",
        "v25",
        "v26"
        );

        w = (width >> 2) << 2;
      }
#else  // gcc
      for (w = 0; w + 3 < width; w += 4) {
        const float *b_ptr = B + (k * stride_w + w);
        float *c_ptr = C + (h * stride_w + w);
        Gemm884(a_ptr, b_ptr, stride_k, stride_w, c_ptr);
      }
#endif  // clang
      if (w < width) {
        const float *a_ptr = A + (h * stride_k + k);
        const float *b_ptr = B + (k * stride_w + w);
        float *c_ptr = C + (h * stride_w + w);
        GemmBlock(a_ptr, b_ptr, 8, 8, width - w, stride_k, stride_w, c_ptr);
      }
    }
    if (k < K) {
      const float *a_ptr = A + (h * stride_k + k);
      const float *b_ptr = B + k * stride_w;
      float *c_ptr = C + h * stride_w;
      GemmBlock(a_ptr,
                b_ptr,
                8,
                K - k,
                width,
                stride_k,
                stride_w,
                c_ptr);
    }
  }
  if (h < height) {
    // TODO(liyin): may use Gemm444
    const float *a_ptr = A + (h * stride_k);
    const float *b_ptr = B;
    float *c_ptr = C + h * stride_w;
    GemmBlock(a_ptr,
              b_ptr,
              height - h,
              K,
              width,
              stride_k,
              stride_w,
              c_ptr);
  }
#else

#if defined(MACE_ENABLE_NEON)  // armv7
  for (h = 0; h + 3 < height; h += 4) {
    for (k = 0; k + 3 < K; k += 4) {
      const float *a_ptr = A + (h * stride_k + k);
      int nw = width >> 2;
      if (nw > 0) {
        // load A
        float32x2_t a00, a01, a10, a11, a20, a21, a30, a31;
        a00 = vld1_f32(a_ptr);
        a01 = vld1_f32(a_ptr + 2);
        a10 = vld1_f32(a_ptr + 1 * stride_k);
        a11 = vld1_f32(a_ptr + 1 * stride_k + 2);
        a20 = vld1_f32(a_ptr + 2 * stride_k);
        a21 = vld1_f32(a_ptr + 2 * stride_k + 2);
        a30 = vld1_f32(a_ptr + 3 * stride_k);
        a31 = vld1_f32(a_ptr + 3 * stride_k + 2);

        const float *b_ptr0 = B + k * stride_w;
        const float *b_ptr1 = B + (k + 1) * stride_w;
        const float *b_ptr2 = B + (k + 2) * stride_w;
        const float *b_ptr3 = B + (k + 3) * stride_w;

        float *c_ptr0 = C + h * stride_w;
        float *c_ptr1 = C + (h + 1) * stride_w;
        float *c_ptr2 = C + (h + 2) * stride_w;
        float *c_ptr3 = C + (h + 3) * stride_w;

        // TODO(liyin): asm v7 prefetch and load optimization
        while (nw--) {
          float32x4_t b0, b1, b2, b3;
          float32x4_t c0, c1, c2, c3;

          c0 = vld1q_f32(c_ptr0);

          b0 = vld1q_f32(b_ptr0);
          b1 = vld1q_f32(b_ptr1);
          b2 = vld1q_f32(b_ptr2);
          b3 = vld1q_f32(b_ptr3);

          c1 = vld1q_f32(c_ptr1);
          c2 = vld1q_f32(c_ptr2);
          c3 = vld1q_f32(c_ptr3);

          c0 = vmlaq_lane_f32(c0, b0, a00, 0);
          c0 = vmlaq_lane_f32(c0, b1, a00, 1);
          c0 = vmlaq_lane_f32(c0, b2, a01, 0);
          c0 = vmlaq_lane_f32(c0, b3, a01, 1);

          vst1q_f32(c_ptr0, c0);

          c1 = vmlaq_lane_f32(c1, b0, a10, 0);
          c1 = vmlaq_lane_f32(c1, b1, a10, 1);
          c1 = vmlaq_lane_f32(c1, b2, a11, 0);
          c1 = vmlaq_lane_f32(c1, b3, a11, 1);

          vst1q_f32(c_ptr1, c1);

          c2 = vmlaq_lane_f32(c2, b0, a20, 0);
          c2 = vmlaq_lane_f32(c2, b1, a20, 1);
          c2 = vmlaq_lane_f32(c2, b2, a21, 0);
          c2 = vmlaq_lane_f32(c2, b3, a21, 1);

          vst1q_f32(c_ptr2, c2);

          c3 = vmlaq_lane_f32(c3, b0, a30, 0);
          c3 = vmlaq_lane_f32(c3, b1, a30, 1);
          c3 = vmlaq_lane_f32(c3, b2, a31, 0);
          c3 = vmlaq_lane_f32(c3, b3, a31, 1);

          vst1q_f32(c_ptr3, c3);

          b_ptr0 += 4;
          b_ptr1 += 4;
          b_ptr2 += 4;
          b_ptr3 += 4;

          c_ptr0 += 4;
          c_ptr1 += 4;
          c_ptr2 += 4;
          c_ptr3 += 4;
        }

        w = (width >> 2) << 2;
      }
      if (w < width) {
        const float *a_ptr = A + (h * stride_k + k);
        const float *b_ptr = B + (k * stride_w + w);
        float *c_ptr = C + (h * stride_w + w);
        GemmBlock(a_ptr, b_ptr, 4, 4, width - w, stride_k, stride_w, c_ptr);
      }
    }
    if (k < K) {
      const float *a_ptr = A + (h * stride_k + k);
      const float *b_ptr = B + k * stride_w;
      float *c_ptr = C + h * stride_w;
      GemmBlock(a_ptr,
                b_ptr,
                4,
                K - k,
                width,
                stride_k,
                stride_w,
                c_ptr);
    }
  }
  if (h < height) {
    const float *a_ptr = A + (h * stride_k);
    const float *b_ptr = B;
    float *c_ptr = C + h * stride_w;
    GemmBlock(a_ptr,
              b_ptr,
              height - h,
              K,
              width,
              stride_k,
              stride_w,
              c_ptr);
  }
#else  // cpu
  GemmBlock(A, B, height, K, width, stride_k, stride_w, C);
#endif  // armv7

#endif  // aarch64
}
}  // namespace

// A: height x K, B: K x width, C: height x width
void Gemm(const float *A,
          const float *B,
          const index_t batch,
          const index_t height,
          const index_t K,
          const index_t width,
          float *C) {
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
  const index_t remain_height = height % block_size;
  const index_t remain_width = width % block_size;
  const index_t remain_k = K % block_size;

#pragma omp parallel for collapse(3)
  for (index_t n = 0; n < batch; ++n) {
    for (index_t bh = 0; bh < block_tile_height; ++bh) {
      for (index_t bw = 0; bw < block_tile_width; ++bw) {
        const float *a_base = A + n * height * K;
        const float *b_base = B + n * K * width;
        float *c_base = C + n * height * width;

        const index_t ih_begin = bh * block_size;
        const index_t ih_end =
          bh * block_size + (bh == block_tile_height - 1 && remain_height > 0
                             ? remain_height : block_size);
        const index_t iw_begin = bw * block_size;
        const index_t iw_end =
          bw * block_size
            + (bw == block_tile_width - 1 && remain_width > 0 ? remain_width
                                                              : block_size);

        for (index_t bk = 0; bk < block_tile_k; ++bk) {
          const index_t ik_begin = bk * block_size;
          const index_t ik_end =
            bk * block_size
              + (bk == block_tile_k - 1 && remain_k > 0 ? remain_k
                                                        : block_size);

          // inside block:
          // calculate C[bh, bw] += A[bh, bk] * B[bk, bw] for one k
          GemmTile(a_base + (ih_begin * K + ik_begin),
                   b_base + (ik_begin * width + iw_begin),
                   ih_end - ih_begin,
                   ik_end - ik_begin,
                   iw_end - iw_begin,
                   K,
                   width,
                   c_base + (ih_begin * width + iw_begin));
        }  // bk
      }  // bw
    }  // bh
  }  // n
}

// A: height x K, B: K x width, C: height x width
void GemmRef(const float *A,
             const float *B,
             const index_t height,
             const index_t K,
             const index_t width,
             float *C) {
  memset(C, 0, sizeof(float) * height * width);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < K; ++k) {
        C[i * width + j] += A[i * K + k] * B[k * width + j];
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
  memset(out_ptr, 0, sizeof(float) * height * batch);
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        out_ptr[h + b * height] += v_ptr[w + b * width] * m_ptr[h * width + w];
      }
    }
  }
}

// M: height x width, Vin: width x 1, Vout: height x 1
void Gemv(const float *m_ptr,
          const float *v_ptr,
          const index_t batch,
          const index_t width,
          const index_t height,
          float *out_ptr) {
#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
  index_t height_d4 = height >> 2;
  index_t width_d4 = width >> 2;
  index_t remain_w = width - (width_d4 << 2);
  index_t remain_h = height - (height_d4 << 2);

  for (index_t b = 0; b < batch; ++b) {
#pragma omp parallel for
    for (index_t h = 0; h < height_d4; ++h) {
      const float *m_ptr0 = m_ptr + h * width * 4;
      const float *m_ptr1 = m_ptr0 + width;
      const float *m_ptr2 = m_ptr1 + width;
      const float *m_ptr3 = m_ptr2 + width;
      const float *v_ptr0 = v_ptr + b * width;
      float *out_ptr0 = out_ptr + h * 4 + b * height;

      float32x4_t vm0, vm1, vm2, vm3;
      float32x4_t vv;

      float32x4_t vsum0 = vdupq_n_f32(0.f);
      float32x4_t vsum1 = vdupq_n_f32(0.f);
      float32x4_t vsum2 = vdupq_n_f32(0.f);
      float32x4_t vsum3 = vdupq_n_f32(0.f);

      for (index_t w = 0; w < width_d4; ++w) {
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
      for (index_t w = 0; w < remain_w; ++w) {
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
    }

    // handle remaining h
    index_t remain_start_height = height_d4 << 2;
#pragma omp parallel for
    for (index_t h = 0; h < remain_h; ++h) {
      float32x4_t vsum0 = vdupq_n_f32(0.f);
      const float *m_ptr0 = m_ptr + (h + remain_start_height) * width;
      const float *v_ptr0 = v_ptr + b * width;
      for (index_t w = 0; w < width_d4; ++w) {
        float32x4_t vm = vld1q_f32(m_ptr0);
        float32x4_t vv = vld1q_f32(v_ptr0);
        vsum0 = vmlaq_f32(vsum0, vm, vv);
        m_ptr0 += 4;
        v_ptr0 += 4;
      }
      float sum = vaddvq_f32(vsum0);
      for (index_t w = 0; w < remain_w; ++w) {
        sum += m_ptr0[0] * v_ptr0[0];
        m_ptr0++;
        v_ptr0++;
      }
      out_ptr[remain_start_height + h + b * height] = sum;
    }
  }
#else
  GemvRef(m_ptr, v_ptr, batch, width, height, out_ptr);
#endif
}

}  // namespace kernels
}  // namespace mace
