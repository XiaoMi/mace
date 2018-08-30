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

#include "mace/kernels/sgemm.h"

#include <memory>

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#if defined(MACE_ENABLE_NEON) && !defined(__aarch64__)
#define vaddvq_f32(v) ((v)[0] + (v)[1] + (v)[2] + (v)[3])
#endif

namespace mace {
namespace kernels {

void SGemm::operator()(const MatrixMap<const float> &lhs,
                       const MatrixMap<const float> &rhs,
                       MatrixMap<float> *result) {
  if (rhs.col() < 16 && lhs.row() >= 16) {
    MatrixMap<const float> lhs_transpose = lhs.transpose();
    MatrixMap<const float> rhs_transpose = rhs.transpose();
    MatrixMap<float> result_transpose = result->transpose();
    return operator()(rhs_transpose, lhs_transpose, &result_transpose);
  }

  if (!packed_ || !lhs.is_const()) {
    PackLhs(lhs, &packed_lhs_);
  }
  if (!packed_ || !rhs.is_const()) {
    PackRhs(rhs, &packed_rhs_);
  }
  packed_ = true;

  PackedBlock<float> packed_result;
  operator()(packed_lhs_,
             packed_rhs_,
             lhs.row(),
             lhs.col(),
             rhs.col(),
             &packed_result);
  UnPack(packed_result, result);
}

#if defined(MACE_ENABLE_NEON)
#if defined(__aarch64__)

// calculate 8 rows, 4 cols for each depth
#define MACE_SGEMM_PART_CAL_R8_C4_D1(D, VD, VDN)      \
  c0 = vfmaq_laneq_f32(c0, b##D, a##VD, 0);           \
  c1 = vfmaq_laneq_f32(c1, b##D, a##VD, 1);           \
  c2 = vfmaq_laneq_f32(c2, b##D, a##VD, 2);           \
  c3 = vfmaq_laneq_f32(c3, b##D, a##VD, 3);           \
  c4 = vfmaq_laneq_f32(c4, b##D, a##VDN, 0);          \
  c5 = vfmaq_laneq_f32(c5, b##D, a##VDN, 1);          \
  c6 = vfmaq_laneq_f32(c6, b##D, a##VDN, 2);          \
  c7 = vfmaq_laneq_f32(c7, b##D, a##VDN, 3);

// calculate 4 rows, 4 cols for each depth
#define MACE_SGEMM_PART_CAL_R4_C4_D1(D)               \
  c0 = vfmaq_laneq_f32(c0, b##D, a##D, 0);            \
  c1 = vfmaq_laneq_f32(c1, b##D, a##D, 1);            \
  c2 = vfmaq_laneq_f32(c2, b##D, a##D, 2);            \
  c3 = vfmaq_laneq_f32(c3, b##D, a##D, 3);

// calculate 4 cols for 8 depths for each row
#define MACE_SGEMM_PART_CAL_R1_C4_D8(R, VR, VRN)      \
  c##R = vfmaq_laneq_f32(c##R, b0, a##VR, 0);         \
  c##R = vfmaq_laneq_f32(c##R, b1, a##VR, 1);         \
  c##R = vfmaq_laneq_f32(c##R, b2, a##VR, 2);         \
  c##R = vfmaq_laneq_f32(c##R, b3, a##VR, 3);         \
  c##R = vfmaq_laneq_f32(c##R, b4, a##VRN, 0);        \
  c##R = vfmaq_laneq_f32(c##R, b5, a##VRN, 1);        \
  c##R = vfmaq_laneq_f32(c##R, b6, a##VRN, 2);        \
  c##R = vfmaq_laneq_f32(c##R, b7, a##VRN, 3);

// calculate 4 cols for 4 depths for each row
#define MACE_SGEMM_PART_CAL_R1_C4_D4(R)               \
  c##R = vfmaq_laneq_f32(c##R, b0, a##R, 0);          \
  c##R = vfmaq_laneq_f32(c##R, b1, a##R, 1);          \
  c##R = vfmaq_laneq_f32(c##R, b2, a##R, 2);          \
  c##R = vfmaq_laneq_f32(c##R, b3, a##R, 3);

// calculate 8 cols for 4 depths for each row
#define MACE_SGEMM_PART_CAL_R1_C8_D4(VR, VRN, R)     \
  c##VR = vfmaq_laneq_f32(c##VR, b0, a##R, 0);       \
  c##VR = vfmaq_laneq_f32(c##VR, b2, a##R, 1);       \
  c##VR = vfmaq_laneq_f32(c##VR, b4, a##R, 2);       \
  c##VR = vfmaq_laneq_f32(c##VR, b6, a##R, 3);       \
  c##VRN = vfmaq_laneq_f32(c##VRN, b1, a##R, 0);     \
  c##VRN = vfmaq_laneq_f32(c##VRN, b3, a##R, 1);     \
  c##VRN = vfmaq_laneq_f32(c##VRN, b5, a##R, 2);     \
  c##VRN = vfmaq_laneq_f32(c##VRN, b7, a##R, 3);

#else

#define MACE_SGEMM_PART_CAL_R8_C4_D1(D, VD, VDN)             \
  c0 = vmlaq_lane_f32(c0, b##D, vget_low_f32(a##VD), 0);     \
  c1 = vmlaq_lane_f32(c1, b##D, vget_low_f32(a##VD), 1);     \
  c2 = vmlaq_lane_f32(c2, b##D, vget_high_f32(a##VD), 0);    \
  c3 = vmlaq_lane_f32(c3, b##D, vget_high_f32(a##VD), 1);    \
  c4 = vmlaq_lane_f32(c4, b##D, vget_low_f32(a##VDN), 0);    \
  c5 = vmlaq_lane_f32(c5, b##D, vget_low_f32(a##VDN), 1);    \
  c6 = vmlaq_lane_f32(c6, b##D, vget_high_f32(a##VDN), 0);   \
  c7 = vmlaq_lane_f32(c7, b##D, vget_high_f32(a##VDN), 1);

#define MACE_SGEMM_PART_CAL_R4_C4_D1(D)                      \
  c0 = vmlaq_lane_f32(c0, b##D, vget_low_f32(a##D), 0);      \
  c1 = vmlaq_lane_f32(c1, b##D, vget_low_f32(a##D), 1);      \
  c2 = vmlaq_lane_f32(c2, b##D, vget_high_f32(a##D), 0);     \
  c3 = vmlaq_lane_f32(c3, b##D, vget_high_f32(a##D), 1);

#define MACE_SGEMM_PART_CAL_R1_C4_D8(R, VR, VRN)             \
  c##R = vmlaq_lane_f32(c##R, b0, vget_low_f32(a##VR), 0);   \
  c##R = vmlaq_lane_f32(c##R, b1, vget_low_f32(a##VR), 1);   \
  c##R = vmlaq_lane_f32(c##R, b2, vget_high_f32(a##VR), 0);  \
  c##R = vmlaq_lane_f32(c##R, b3, vget_high_f32(a##VR), 1);  \
  c##R = vmlaq_lane_f32(c##R, b4, vget_low_f32(a##VRN), 0);  \
  c##R = vmlaq_lane_f32(c##R, b5, vget_low_f32(a##VRN), 1);  \
  c##R = vmlaq_lane_f32(c##R, b6, vget_high_f32(a##VRN), 0); \
  c##R = vmlaq_lane_f32(c##R, b7, vget_high_f32(a##VRN), 1);

#define MACE_SGEMM_PART_CAL_R1_C4_D4(R)                      \
  c##R = vmlaq_lane_f32(c##R, b0, vget_low_f32(a##R), 0);    \
  c##R = vmlaq_lane_f32(c##R, b1, vget_low_f32(a##R), 1);    \
  c##R = vmlaq_lane_f32(c##R, b2, vget_high_f32(a##R), 0);   \
  c##R = vmlaq_lane_f32(c##R, b3, vget_high_f32(a##R), 1);

#endif  // __aarch64__
#endif  // MACE_ENABLE_NEON

void SGemm::operator()(const PackedBlock<float> &lhs,
                       const PackedBlock<float> &rhs,
                       const index_t height,
                       const index_t depth,
                       const index_t width,
                       PackedBlock<float> *result) {
  result->tensor()->Resize({height * width});
  const float *lhs_data = lhs.data();
  const float *rhs_data = rhs.data();
  float *result_data = result->mutable_data();

#if defined(MACE_ENABLE_NEON)
  const index_t block_w = width >> 2;
  const index_t remain_w = width - (block_w << 2);
#else
  const index_t remain_w = width;
#endif

#if defined(MACE_ENABLE_NEON)
  // TODO(liyin): collapse loop

  // w: 4
#pragma omp parallel for
  for (index_t bw = 0; bw < block_w; ++bw) {
    index_t remain_h = height;
    index_t block_h = 0;

    const float *lhs_ptr = lhs_data;
    float *res_ptr = result_data + height * (bw << 2);

#if defined(__aarch64__)
    block_h = remain_h >> 3;
    remain_h -= (block_h << 3);

    // h: 8
    for (index_t bh = 0; bh < block_h; ++bh) {
      const float *rhs_ptr = rhs_data + depth * (bw << 2);

      index_t remain_d = depth;
      index_t block_d = remain_d >> 3;
      remain_d -= (block_d << 3);

      float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;
      c0 = vdupq_n_f32(0.f);
      c1 = vdupq_n_f32(0.f);
      c2 = vdupq_n_f32(0.f);
      c3 = vdupq_n_f32(0.f);
      c4 = vdupq_n_f32(0.f);
      c5 = vdupq_n_f32(0.f);
      c6 = vdupq_n_f32(0.f);
      c7 = vdupq_n_f32(0.f);

      // d: 8
      for (index_t bd = 0; bd < block_d; ++bd) {
        // 8.8.4
        float32x4_t a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13,
            a14, a15;
        float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;

        a0 = vld1q_f32(lhs_ptr);
        a1 = vld1q_f32(lhs_ptr + 4);
        a2 = vld1q_f32(lhs_ptr + 8);
        a3 = vld1q_f32(lhs_ptr + 12);
        a4 = vld1q_f32(lhs_ptr + 16);
        a5 = vld1q_f32(lhs_ptr + 20);
        a6 = vld1q_f32(lhs_ptr + 24);
        a7 = vld1q_f32(lhs_ptr + 28);
        a8 = vld1q_f32(lhs_ptr + 32);
        a9 = vld1q_f32(lhs_ptr + 36);
        a10 = vld1q_f32(lhs_ptr + 40);
        a11 = vld1q_f32(lhs_ptr + 44);
        a12 = vld1q_f32(lhs_ptr + 48);
        a13 = vld1q_f32(lhs_ptr + 52);
        a14 = vld1q_f32(lhs_ptr + 56);
        a15 = vld1q_f32(lhs_ptr + 60);

        b0 = vld1q_f32(rhs_ptr);
        b1 = vld1q_f32(rhs_ptr + 4);
        b2 = vld1q_f32(rhs_ptr + 8);
        b3 = vld1q_f32(rhs_ptr + 12);
        b4 = vld1q_f32(rhs_ptr + 16);
        b5 = vld1q_f32(rhs_ptr + 20);
        b6 = vld1q_f32(rhs_ptr + 24);
        b7 = vld1q_f32(rhs_ptr + 28);

        MACE_SGEMM_PART_CAL_R8_C4_D1(0, 0, 1);  // d = 1
        MACE_SGEMM_PART_CAL_R8_C4_D1(1, 2, 3);  // d = 2
        MACE_SGEMM_PART_CAL_R8_C4_D1(2, 4, 5);
        MACE_SGEMM_PART_CAL_R8_C4_D1(3, 6, 7);
        MACE_SGEMM_PART_CAL_R8_C4_D1(4, 8, 9);
        MACE_SGEMM_PART_CAL_R8_C4_D1(5, 10, 11);
        MACE_SGEMM_PART_CAL_R8_C4_D1(6, 12, 13);
        MACE_SGEMM_PART_CAL_R8_C4_D1(7, 14, 15);

        lhs_ptr += 64;
        rhs_ptr += 32;
      }

      block_d = remain_d >> 2;
      remain_d -= (block_d << 2);

      // d: 4
      for (index_t bd = 0; bd < block_d; ++bd) {
        // 8.4.4
        float32x4_t a0, a1, a2, a3, a4, a5, a6, a7;
        float32x4_t b0, b1, b2, b3;

        a0 = vld1q_f32(lhs_ptr);
        a1 = vld1q_f32(lhs_ptr + 4);
        a2 = vld1q_f32(lhs_ptr + 8);
        a3 = vld1q_f32(lhs_ptr + 12);
        a4 = vld1q_f32(lhs_ptr + 16);
        a5 = vld1q_f32(lhs_ptr + 20);
        a6 = vld1q_f32(lhs_ptr + 24);
        a7 = vld1q_f32(lhs_ptr + 28);

        b0 = vld1q_f32(rhs_ptr);
        b1 = vld1q_f32(rhs_ptr + 4);
        b2 = vld1q_f32(rhs_ptr + 8);
        b3 = vld1q_f32(rhs_ptr + 12);

        MACE_SGEMM_PART_CAL_R8_C4_D1(0, 0, 1);  // d = 1
        MACE_SGEMM_PART_CAL_R8_C4_D1(1, 2, 3);  // d = 2
        MACE_SGEMM_PART_CAL_R8_C4_D1(2, 4, 5);
        MACE_SGEMM_PART_CAL_R8_C4_D1(3, 6, 7);

        lhs_ptr += 32;
        rhs_ptr += 16;
      }

      // TODO(liyin): handle remain by each case
      // d: remain
      for (index_t d = 0; d < remain_d; ++d) {
        // 8.1.4
        float32x4_t a0, a1;
        float32x4_t b0;

        a0 = vld1q_f32(lhs_ptr);
        a1 = vld1q_f32(lhs_ptr + 4);

        b0 = vld1q_f32(rhs_ptr);

        MACE_SGEMM_PART_CAL_R8_C4_D1(0, 0, 1);  // d = 1

        lhs_ptr += 8;
        rhs_ptr += 4;
      }

      vst1q_f32(res_ptr, c0);
      vst1q_f32(res_ptr + 4, c1);
      vst1q_f32(res_ptr + 8, c2);
      vst1q_f32(res_ptr + 12, c3);
      vst1q_f32(res_ptr + 16, c4);
      vst1q_f32(res_ptr + 20, c5);
      vst1q_f32(res_ptr + 24, c6);
      vst1q_f32(res_ptr + 28, c7);

      res_ptr += 32;
    }  // bh: 8
#endif  // __aarch64__

    // h: 4
    block_h = remain_h >> 2;
    remain_h -= (block_h << 2);

    for (index_t bh = 0; bh < block_h; ++bh) {
      const float *rhs_ptr = rhs_data + depth * (bw << 2);

      index_t remain_d = depth;
      index_t block_d = 0;

      float32x4_t c0, c1, c2, c3;
      c0 = vdupq_n_f32(0.f);
      c1 = vdupq_n_f32(0.f);
      c2 = vdupq_n_f32(0.f);
      c3 = vdupq_n_f32(0.f);

#if defined(__aarch64__)
      block_d = remain_d >> 3;
      remain_d -= (block_d << 3);

      // d: 8
      for (index_t bd = 0; bd < block_d; ++bd) {
        // 4.8.4
        float32x4_t a0, a1, a2, a3, a4, a5, a6, a7;
        float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;

        a0 = vld1q_f32(lhs_ptr);
        a1 = vld1q_f32(lhs_ptr + 4);
        a2 = vld1q_f32(lhs_ptr + 8);
        a3 = vld1q_f32(lhs_ptr + 12);
        a4 = vld1q_f32(lhs_ptr + 16);
        a5 = vld1q_f32(lhs_ptr + 20);
        a6 = vld1q_f32(lhs_ptr + 24);
        a7 = vld1q_f32(lhs_ptr + 28);

        b0 = vld1q_f32(rhs_ptr);
        b1 = vld1q_f32(rhs_ptr + 4);
        b2 = vld1q_f32(rhs_ptr + 8);
        b3 = vld1q_f32(rhs_ptr + 12);
        b4 = vld1q_f32(rhs_ptr + 16);
        b5 = vld1q_f32(rhs_ptr + 20);
        b6 = vld1q_f32(rhs_ptr + 24);
        b7 = vld1q_f32(rhs_ptr + 28);

        MACE_SGEMM_PART_CAL_R4_C4_D1(0);  // d = 1
        MACE_SGEMM_PART_CAL_R4_C4_D1(1);  // d = 2
        MACE_SGEMM_PART_CAL_R4_C4_D1(2);
        MACE_SGEMM_PART_CAL_R4_C4_D1(3);
        MACE_SGEMM_PART_CAL_R4_C4_D1(4);
        MACE_SGEMM_PART_CAL_R4_C4_D1(5);
        MACE_SGEMM_PART_CAL_R4_C4_D1(6);
        MACE_SGEMM_PART_CAL_R4_C4_D1(7);

        lhs_ptr += 32;
        rhs_ptr += 32;
      }
#endif  // __aarch64__

      block_d = remain_d >> 2;
      remain_d -= (block_d << 2);

      // d: 4
      for (index_t bd = 0; bd < block_d; ++bd) {
        // 4.4.4
        float32x4_t a0, a1, a2, a3;
        float32x4_t b0, b1, b2, b3;

        a0 = vld1q_f32(lhs_ptr);
        a1 = vld1q_f32(lhs_ptr + 4);
        a2 = vld1q_f32(lhs_ptr + 8);
        a3 = vld1q_f32(lhs_ptr + 12);

        b0 = vld1q_f32(rhs_ptr);
        b1 = vld1q_f32(rhs_ptr + 4);
        b2 = vld1q_f32(rhs_ptr + 8);
        b3 = vld1q_f32(rhs_ptr + 12);

        MACE_SGEMM_PART_CAL_R4_C4_D1(0);  // d = 1
        MACE_SGEMM_PART_CAL_R4_C4_D1(1);  // d = 2
        MACE_SGEMM_PART_CAL_R4_C4_D1(2);
        MACE_SGEMM_PART_CAL_R4_C4_D1(3);

        lhs_ptr += 16;
        rhs_ptr += 16;
      }

      // d: remain
      for (index_t d = 0; d < remain_d; ++d) {
        // 4.1.4
        float32x4_t a0;
        float32x4_t b0;

        a0 = vld1q_f32(lhs_ptr);

        b0 = vld1q_f32(rhs_ptr);

        MACE_SGEMM_PART_CAL_R4_C4_D1(0);  // d = 1

        lhs_ptr += 4;
        rhs_ptr += 4;
      }
      vst1q_f32(res_ptr, c0);
      vst1q_f32(res_ptr + 4, c1);
      vst1q_f32(res_ptr + 8, c2);
      vst1q_f32(res_ptr + 12, c3);

      res_ptr += 16;
    }  // bh: 4

    // h: 1
    for (index_t h = 0; h < remain_h; ++h) {
      const float *rhs_ptr = rhs_data + depth * (bw << 2);

      index_t remain_d = depth;
      index_t block_d = 0;

      float32x4_t c0 = vdupq_n_f32(0.f);

#if defined(__aarch64__)
      block_d = remain_d >> 3;
      remain_d -= (block_d << 3);

      // d: 8
      for (index_t bd = 0; bd < block_d; ++bd) {
        // 1.8.4
        float32x4_t a0, a1;
        float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;

        a0 = vld1q_f32(lhs_ptr);
        a1 = vld1q_f32(lhs_ptr + 4);

        b0 = vld1q_f32(rhs_ptr);
        b1 = vld1q_f32(rhs_ptr + 4);
        b2 = vld1q_f32(rhs_ptr + 8);
        b3 = vld1q_f32(rhs_ptr + 12);
        b4 = vld1q_f32(rhs_ptr + 16);
        b5 = vld1q_f32(rhs_ptr + 20);
        b6 = vld1q_f32(rhs_ptr + 24);
        b7 = vld1q_f32(rhs_ptr + 28);

        MACE_SGEMM_PART_CAL_R1_C4_D8(0, 0, 1);

        lhs_ptr += 8;
        rhs_ptr += 32;
      }
#endif  // __aarch64__

      block_d = remain_d >> 2;
      remain_d -= (block_d << 2);

      // d: 4
      for (index_t bd = 0; bd < block_d; ++bd) {
        // 1.4.4
        float32x4_t a0;
        float32x4_t b0, b1, b2, b3;

        a0 = vld1q_f32(lhs_ptr);

        b0 = vld1q_f32(rhs_ptr);
        b1 = vld1q_f32(rhs_ptr + 4);
        b2 = vld1q_f32(rhs_ptr + 8);
        b3 = vld1q_f32(rhs_ptr + 12);

        MACE_SGEMM_PART_CAL_R1_C4_D4(0);

        lhs_ptr += 4;
        rhs_ptr += 16;
      }

      // d: remain
      float s0 = 0;
      float s1 = 0;
      float s2 = 0;
      float s3 = 0;
      for (index_t d = 0; d < remain_d; ++d) {
        // 1.1.4
        s0 += lhs_ptr[0] * rhs_ptr[0];
        s1 += lhs_ptr[0] * rhs_ptr[1];
        s2 += lhs_ptr[0] * rhs_ptr[2];
        s3 += lhs_ptr[0] * rhs_ptr[3];
        lhs_ptr += 1;
        rhs_ptr += 4;
      }
      float32x4_t c0_remain = {s0, s1, s2, s3};
      c0 += c0_remain;

      vst1q_f32(res_ptr, c0);
      res_ptr += 4;
    }  // bh: remain
  }  // bw

#endif  // MACE_ENABLE_NEON

  // ========================== remain width ===========================

  result_data += (width - remain_w) * height;
  rhs_data += (width - remain_w) * depth;

  // w: 1
#pragma omp parallel for
  for (index_t bw = 0; bw < remain_w; ++bw) {
    index_t remain_h = height;
    index_t block_h = 0;

    const float *lhs_ptr = lhs_data;
    float *res_ptr = result_data + height * bw;

#if defined(MACE_ENABLE_NEON)
#if defined(__aarch64__)
    block_h = remain_h >> 3;
    remain_h -= (block_h << 3);

    // h: 8
    for (index_t bh = 0; bh < block_h; ++bh) {
      const float *rhs_ptr = rhs_data + depth * bw;

      index_t remain_d = depth;

      float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;
      c0 = vdupq_n_f32(0.f);
      c1 = vdupq_n_f32(0.f);

      index_t block_d = remain_d >> 2;
      remain_d -= (block_d << 2);

      // d: 4
      for (index_t bd = 0; bd < block_d; ++bd) {
        // 8.4.1
        float32x4_t b0, b1, b2, b3, b4, b5, b6, b7;
        float32x4_t a0;

        b0 = vld1q_f32(lhs_ptr);
        b1 = vld1q_f32(lhs_ptr + 4);
        b2 = vld1q_f32(lhs_ptr + 8);
        b3 = vld1q_f32(lhs_ptr + 12);
        b4 = vld1q_f32(lhs_ptr + 16);
        b5 = vld1q_f32(lhs_ptr + 20);
        b6 = vld1q_f32(lhs_ptr + 24);
        b7 = vld1q_f32(lhs_ptr + 28);

        a0 = vld1q_f32(rhs_ptr);

        MACE_SGEMM_PART_CAL_R1_C8_D4(0, 1, 0);

        lhs_ptr += 32;
        rhs_ptr += 4;
      }

      // d: remain
      for (index_t d = 0; d < remain_d; ++d) {
        // 8.1.1
        float32x4_t b0, b1;

        b0 = vld1q_f32(lhs_ptr);
        b1 = vld1q_f32(lhs_ptr + 4);

        c0 += b0 * rhs_ptr[0];
        c1 += b1 * rhs_ptr[0];

        lhs_ptr += 8;
        rhs_ptr += 1;
      }

      vst1q_f32(res_ptr, c0);
      vst1q_f32(res_ptr + 4, c1);

      res_ptr += 8;
    }  // bh: 8
#endif

    // h: 4
    block_h = remain_h >> 2;
    remain_h -= (block_h << 2);

    for (index_t bh = 0; bh < block_h; ++bh) {
      const float *rhs_ptr = rhs_data + depth * bw;

      index_t remain_d = depth;
      index_t block_d = 0;

      float32x4_t c0 = vdupq_n_f32(0.f);

      block_d = remain_d >> 2;
      remain_d -= (block_d << 2);

      // d: 4
      for (index_t bd = 0; bd < block_d; ++bd) {
        // 4.4.1
        float32x4_t b0, b1, b2, b3;
        float32x4_t a0;

        b0 = vld1q_f32(lhs_ptr);
        b1 = vld1q_f32(lhs_ptr + 4);
        b2 = vld1q_f32(lhs_ptr + 8);
        b3 = vld1q_f32(lhs_ptr + 12);

        a0 = vld1q_f32(rhs_ptr);

        MACE_SGEMM_PART_CAL_R1_C4_D4(0);

        lhs_ptr += 16;
        rhs_ptr += 4;
      }

      // d: remain
      for (index_t d = 0; d < remain_d; ++d) {
        // 4.1.1
        float32x4_t b0, b1;

        b0 = vld1q_f32(lhs_ptr);

        c0 += b0 * rhs_ptr[0];

        lhs_ptr += 4;
        rhs_ptr += 1;
      }
      vst1q_f32(res_ptr, c0);

      res_ptr += 4;
    }  // bh: 4

#endif  // MACE_ENABLE_NEON

    // h: 1
    for (index_t h = 0; h < remain_h; ++h) {
      const float *rhs_ptr = rhs_data + depth * bw;

      index_t remain_d = depth;
      index_t block_d = 0;

      float sum = 0.f;

#if defined(MACE_ENABLE_NEON)
      float32x4_t c0;
      c0 = vdupq_n_f32(0.f);

      block_d = remain_d >> 2;
      remain_d -= (block_d << 2);

      // d: 4
      for (index_t bd = 0; bd < block_d; ++bd) {
        // 1.4.1
        float32x4_t a0;
        float32x4_t b0;

        a0 = vld1q_f32(lhs_ptr);
        b0 = vld1q_f32(rhs_ptr);

        c0 = vmlaq_f32(c0, a0, b0);

        lhs_ptr += 4;
        rhs_ptr += 4;
      }
      sum = vaddvq_f32(c0);
#endif  // MACE_ENABLE_NEON

      // d: remain
      for (index_t d = 0; d < remain_d; ++d) {
        // 1.1.1
        sum += lhs_ptr[0] * rhs_ptr[0];
        lhs_ptr += 1;
        rhs_ptr += 1;
      }

      *res_ptr = sum;
      ++res_ptr;
    }  // bh: remain
  }  // bw
}

void SGemm::PackLhs(const MatrixMap<const float> &lhs,
                    PackedBlock<float> *packed_block) {
  Pack(lhs, PackOrder::ColMajor, packed_block);
}

void SGemm::PackRhs(const MatrixMap<const float> &rhs,
                    PackedBlock<float> *packed_block) {
  Pack(rhs, PackOrder::RowMajor, packed_block);
}

void SGemm::UnPack(const PackedBlock<float> &packed_result,
                   MatrixMap<float> *matrix_map) {
  MACE_CHECK_NOTNULL(matrix_map);

  const index_t height = matrix_map->row();
  const index_t width = matrix_map->col();
  auto packed_data = packed_result.data();
  auto unpacked_data = matrix_map->data();

  if (matrix_map->major() == Major::RowMajor) {
    // This is for non-transposed result
    index_t w = 0;
#if defined(MACE_ENABLE_NEON)
#pragma omp parallel for
    for (index_t iw = w; iw <= width - 4; iw += 4) {
      const float *packed_data_ptr = packed_data + iw * height;
      float *unpacked_data_ptr = unpacked_data + iw;
      for (index_t h = 0; h < height; ++h) {
        const index_t packed_offset = h * 4;
        const index_t unpacked_offset = h * width;
        float32x4_t vs = vld1q_f32(packed_data_ptr + packed_offset);
        vst1q_f32(unpacked_data_ptr + unpacked_offset, vs);
      }
    }
    w += (width - w) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t iw = w; iw < width; ++iw) {
      const float *packed_data_ptr = packed_data + iw * height;
      float *unpacked_data_ptr = unpacked_data + iw;
      for (index_t h = 0; h < height; ++h) {
        unpacked_data_ptr[h * width] = packed_data_ptr[h];
      }
    }
  } else {
    // This is for transposed result
    index_t w = 0;
#if defined(MACE_ENABLE_NEON)
#pragma omp parallel for
    for (index_t iw = w; iw <= width - 4; iw += 4) {
      const float *packed_data_ptr = packed_data + iw * height;
      float *unpacked_data_ptr = unpacked_data + iw * height;
      for (index_t h = 0; h < height; ++h) {
        const index_t packed_offset = h * 4;
        const index_t unpacked_offset = h;
        float32x4_t vs = vld1q_f32(packed_data_ptr + packed_offset);
        unpacked_data_ptr[unpacked_offset] = vs[0];
        unpacked_data_ptr[unpacked_offset + height] = vs[1];
        unpacked_data_ptr[unpacked_offset + 2 * height] = vs[2];
        unpacked_data_ptr[unpacked_offset + 3 * height] = vs[3];
      }
    }
    w += (width - w) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t iw = w; iw < width; ++iw) {
      std::copy_n(
          packed_data + iw * height, height, unpacked_data + iw * height);
    }
  }
}

void SGemm::Pack(const MatrixMap<const float> &src,
                 const PackOrder order,
                 PackedBlock<float> *packed_block) {
  MACE_CHECK_NOTNULL(packed_block);

  const index_t height = src.row();
  const index_t width = src.col();
  packed_block->tensor()->Resize({height * width});
  auto src_data = src.data();
  auto packed_data = packed_block->mutable_data();

  if (src.major() == Major::RowMajor && order == PackOrder::ColMajor) {
    // This is for packing no-transpose lhs.
    index_t h = 0;
#if defined(MACE_ENABLE_NEON)
#if defined(__aarch64__)
#pragma omp parallel for
    for (index_t ih = h; ih <= height - 8; ih += 8) {
      const float *src_data_ptr = src_data + ih * width;
      float *packed_data_ptr = packed_data + ih * width;
      for (index_t w = 0; w < width; ++w) {
        const index_t src_offset = w;
        const index_t packed_offset = w * 8;
        float32x4_t vs0 = {src_data_ptr[src_offset],
                           src_data_ptr[src_offset + width],
                           src_data_ptr[src_offset + 2 * width],
                           src_data_ptr[src_offset + 3 * width]};
        float32x4_t vs1 = {src_data_ptr[src_offset + 4 * width],
                           src_data_ptr[src_offset + 5 * width],
                           src_data_ptr[src_offset + 6 * width],
                           src_data_ptr[src_offset + 7 * width]};
        vst1q_f32(packed_data_ptr + packed_offset, vs0);
        vst1q_f32(packed_data_ptr + packed_offset + 4, vs1);
      }
    }
    h += (height - h) / 8 * 8;
#endif
#pragma omp parallel for
    for (index_t ih = h; ih <= height - 4; ih += 4) {
      const float *src_data_ptr = src_data + ih * width;
      float *packed_data_ptr = packed_data + ih * width;
      for (index_t w = 0; w < width; ++w) {
        const index_t src_offset = w;
        const index_t packed_offset = w * 4;
        float32x4_t vs = {src_data_ptr[src_offset],
                          src_data_ptr[src_offset + width],
                          src_data_ptr[src_offset + 2 * width],
                          src_data_ptr[src_offset + 3 * width]};
        vst1q_f32(packed_data_ptr + packed_offset, vs);
      }
    }
    h += (height - h) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t ih = h; ih < height; ++ih) {
      std::copy_n(src_data + ih * width, width, packed_data + ih * width);
    }
  } else if (src.major() == Major::ColMajor && order == PackOrder::ColMajor) {
    // This is for packing transpose-needed lhs.
    index_t h = 0;
#if defined(MACE_ENABLE_NEON)
#if defined(__aarch64__)
#pragma omp parallel for
    for (index_t ih = h; ih <= height - 8; ih += 8) {
      const float *src_data_ptr = src_data + ih;
      float *packed_data_ptr = packed_data + ih * width;
      for (index_t w = 0; w < width; ++w) {
        const index_t src_offset = w * height;
        const index_t packed_offset = w * 8;
        float32x4_t vs0 = vld1q_f32(src_data_ptr + src_offset);
        float32x4_t vs1 = vld1q_f32(src_data_ptr + src_offset + 4);
        vst1q_f32(packed_data_ptr + packed_offset, vs0);
        vst1q_f32(packed_data_ptr + packed_offset + 4, vs1);
      }
    }
    h += (height - h) / 8 * 8;
#endif
#pragma omp parallel for
    for (index_t ih = h; ih <= height - 4; ih += 4) {
      const float *src_data_ptr = src_data + ih;
      float *packed_data_ptr = packed_data + ih * width;
      for (index_t w = 0; w < width; ++w) {
        const index_t src_offset = w * height;
        const index_t packed_offset = w * 4;
        float32x4_t vs = vld1q_f32(src_data_ptr + src_offset);
        vst1q_f32(packed_data_ptr + packed_offset, vs);
      }
    }
    h += (height - h) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t ih = h; ih < height; ++ih) {
      const float *src_data_ptr = src_data + ih;
      float *packed_data_ptr = packed_data + ih * width;
      for (index_t w = 0; w < width; ++w) {
        packed_data_ptr[w] = src_data_ptr[w * height];
      }
    }
  } else if (src.major() == Major::RowMajor && order == PackOrder::RowMajor) {
    // This is for packing no-transpose rhs.
    index_t w = 0;
#if defined(MACE_ENABLE_NEON)
#pragma omp parallel for
    for (index_t iw = w; iw <= width - 4; iw += 4) {
      const float *src_data_ptr = src_data + iw;
      float *packed_data_ptr = packed_data + iw * height;
      for (index_t h = 0; h < height; ++h) {
        const index_t src_offset = h * width;
        const index_t packed_offset = h * 4;
        float32x4_t vs = vld1q_f32(src_data_ptr + src_offset);
        vst1q_f32(packed_data_ptr + packed_offset, vs);
      }
    }
    w += (width - w) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t iw = w; iw < width; ++iw) {
      const float *src_data_ptr = src_data + iw;
      float *packed_data_ptr = packed_data + iw * height;
      for (index_t h = 0; h < height; ++h) {
        packed_data_ptr[h] = src_data_ptr[h * width];
      }
    }
  } else if (src.major() == Major::ColMajor && order == PackOrder::RowMajor) {
    // This is for packing transpose-needed rhs.
    index_t w = 0;
#if defined(MACE_ENABLE_NEON)
#pragma omp parallel for
    for (index_t iw = w; iw <= width - 4; iw += 4) {
      const float *src_data_ptr = src_data + iw * height;
      float *packed_data_ptr = packed_data + iw * height;
      for (index_t h = 0; h < height; ++h) {
        const index_t src_offset = h;
        const index_t packed_offset = h * 4;
        float32x4_t vs = {src_data_ptr[src_offset],
                          src_data_ptr[src_offset + height],
                          src_data_ptr[src_offset + 2 * height],
                          src_data_ptr[src_offset + 3 * height]};
        vst1q_f32(packed_data_ptr + packed_offset, vs);
      }
    }
    w += (width - w) / 4 * 4;
#endif
#pragma omp parallel for
    for (index_t iw = w; iw < width; ++iw) {
      std::copy_n(src_data + iw * height, height, packed_data + iw * height);
    }
  }
}

}  // namespace kernels
}  // namespace mace
