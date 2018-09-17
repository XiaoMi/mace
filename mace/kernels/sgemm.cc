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

#include <memory>

#include "mace/kernels/sgemm.h"
#include "mace/core/runtime/cpu/cpu_runtime.h"


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
                       MatrixMap<float> *result,
                       ScratchBuffer *scratch_buffer) {
  if (rhs.col() < lhs.row()) {
    MatrixMap<const float> lhs_transpose = lhs.transpose();
    MatrixMap<const float> rhs_transpose = rhs.transpose();
    MatrixMap<float> result_transpose = result->transpose();
    return operator()(rhs_transpose,
                      lhs_transpose,
                      &result_transpose,
                      scratch_buffer);
  }

  if (scratch_buffer != nullptr) {
    index_t total_size = result->size();
    if (!lhs.is_const()) {
      total_size += lhs.size();
    }
    if (!rhs.is_const()) {
      total_size += rhs.size();
    }
    scratch_buffer->GrowSize(total_size * sizeof(float));

    if (!lhs.is_const()) {
      packed_lhs_.reset(new Tensor(scratch_buffer->Scratch(
          lhs.size() * sizeof(float)), DT_FLOAT));
    }
    if (!rhs.is_const()) {
      packed_rhs_.reset(new Tensor(scratch_buffer->Scratch(
          rhs.size() * sizeof(float)), DT_FLOAT));
    }
    packed_result_.reset(new Tensor(scratch_buffer->Scratch(
        result->size() * sizeof(float)), DT_FLOAT));
  }

  if (packed_lhs_.get() == nullptr) {
    packed_lhs_.reset(new Tensor(GetCPUAllocator(), DT_FLOAT));
    packed_lhs_->Resize({lhs.size()});
  }
  if (packed_rhs_.get() == nullptr) {
    packed_rhs_.reset(new Tensor(GetCPUAllocator(), DT_FLOAT));
    packed_rhs_->Resize({rhs.size()});
  }
  if (packed_result_.get() == nullptr) {
    packed_result_.reset(new Tensor(GetCPUAllocator(), DT_FLOAT));
    packed_result_->Resize({result->size()});
  }

  if (!lhs.is_const() || !packed_) {
    PackLhs(lhs, packed_lhs_.get());
  }
  if (!rhs.is_const() || !packed_) {
    PackRhs(rhs, packed_rhs_.get());
  }
  packed_ = true;

  RunInternal(*packed_lhs_,
              *packed_rhs_,
              lhs.batch(),
              lhs.row(),
              lhs.col(),
              rhs.col(),
              packed_result_.get());

  UnPack(*packed_result_, result);
}

void SGemm::Run(const float *A,
                const float *B,
                const index_t batch,
                const index_t height_a,
                const index_t width_a,
                const index_t height_b,
                const index_t width_b,
                const bool transpose_a,
                const bool transpose_b,
                const bool is_a_weight,
                const bool is_b_weight,
                float *C,
                ScratchBuffer *scratch_buffer) {
  index_t height_c = height_a;
  index_t width_c = width_b;
  if (transpose_a) {
    height_c = width_a;
  }
  if (transpose_b) {
    width_c = height_b;
  }

  MatrixMap<const float> matrix_a =
      MatrixMap<const float>(batch,
                             height_a,
                             width_a,
                             kernels::RowMajor,
                             A,
                             is_a_weight);
  MatrixMap<const float> matrix_b =
      kernels::MatrixMap<const float>(batch,
                                      height_b,
                                      width_b,
                                      kernels::RowMajor,
                                      B,
                                      is_b_weight);
  if (transpose_a) {
    matrix_a = matrix_a.transpose();
  }
  if (transpose_b) {
    matrix_b = matrix_b.transpose();
  }
  MatrixMap<float> matrix_c(batch, height_c, width_c, kernels::RowMajor, C);
  operator()(matrix_a, matrix_b, &matrix_c, scratch_buffer);
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

void SGemm::RunInternal(const PackedBlock &lhs,
                        const PackedBlock &rhs,
                        const index_t batch,
                        const index_t height,
                        const index_t depth,
                        const index_t width,
                        PackedBlock *result) {
  const float *lhs_data = lhs.data<float>();
  const float *rhs_data = rhs.data<float>();
  float *result_data = result->mutable_data<float>();

#define MACE_SGEMM_RUN_PER_BATCH                      \
  for (index_t b = 0; b < batch; ++b) {               \
    RunPerBatch(lhs_data + b * height * depth,        \
                rhs_data + b * depth * width,         \
                height,                               \
                depth,                                \
                width,                                \
                result_data + b * height * width);    \
  }

  if (batch >= MaceOpenMPThreadCount) {
#pragma omp parallel for
    MACE_SGEMM_RUN_PER_BATCH
  } else {
    MACE_SGEMM_RUN_PER_BATCH
  }

#undef MACE_SGEMM_RUN_PER_BATCH
}

void SGemm::RunPerBatch(const float *lhs_data,
                        const float *rhs_data,
                        const index_t height,
                        const index_t depth,
                        const index_t width,
                        float *result_data) {
#if defined(MACE_ENABLE_NEON)
  const index_t block_w = width >> 2;
  const index_t remain_w = width - (block_w << 2);
#else
  const index_t remain_w = width;
#endif

#if defined(MACE_ENABLE_NEON)
  // TODO(liyin): make better use l2(l1) cache, try to fit as much lhs data as
  // as possible to cache, by tiling lhs by height and rhs by width.

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

      // d: 8
      block_d = remain_d >> 3;
      remain_d -= (block_d << 3);

#if defined(__aarch64__)
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
#else  // arm v7
      // 4.8.4
      if (block_d > 0) {
        asm volatile(
          "0: \n"

          "vld1.f32 {d0-d1}, [%[lhs_ptr]]! \n"
          "vld1.f32 {d2-d3}, [%[lhs_ptr]]! \n"
          "vld1.f32 {d4-d5}, [%[lhs_ptr]]! \n"

          "vld1.f32 {d20-d21}, [%[rhs_ptr]]! \n"
          "vld1.f32 {d22-d23}, [%[rhs_ptr]]! \n"
          "vld1.f32 {d24-d25}, [%[rhs_ptr]]! \n"

          "vmla.f32 %[c0], q10, d0[0] \n"
          "vmla.f32 %[c1], q10, d0[1] \n"
          "vmla.f32 %[c2], q10, d1[0] \n"
          "vmla.f32 %[c3], q10, d1[1] \n"

          "vld1.f32 {d6-d7}, [%[lhs_ptr]]! \n"
          "vld1.f32 {d26-d27}, [%[rhs_ptr]]! \n"

          "vmla.f32 %[c0], q11, d2[0] \n"
          "vmla.f32 %[c1], q11, d2[1] \n"
          "vmla.f32 %[c2], q11, d3[0] \n"
          "vmla.f32 %[c3], q11, d3[1] \n"

          "vld1.f32 {d8-d9}, [%[lhs_ptr]]! \n"
          "vld1.f32 {d28-d29}, [%[rhs_ptr]]! \n"

          "vmla.f32 %[c0], q12, d4[0] \n"
          "vmla.f32 %[c1], q12, d4[1] \n"
          "vmla.f32 %[c2], q12, d5[0] \n"
          "vmla.f32 %[c3], q12, d5[1] \n"

          "vld1.f32 {d10-d11}, [%[lhs_ptr]]! \n"
          "vld1.f32 {d30-d31}, [%[rhs_ptr]]! \n"

          "vmla.f32 %[c0], q13, d6[0] \n"
          "vmla.f32 %[c1], q13, d6[1] \n"
          "vmla.f32 %[c2], q13, d7[0] \n"
          "vmla.f32 %[c3], q13, d7[1] \n"

          "vld1.f32 {d0-d1}, [%[lhs_ptr]]! \n"
          "vld1.f32 {d2-d3}, [%[lhs_ptr]]! \n"

          "vld1.f32 {d20-d21}, [%[rhs_ptr]]! \n"
          "vld1.f32 {d22-d23}, [%[rhs_ptr]]! \n"

          "vmla.f32 %[c0], q14, d8[0] \n"
          "vmla.f32 %[c1], q14, d8[1] \n"
          "vmla.f32 %[c2], q14, d9[0] \n"
          "vmla.f32 %[c3], q14, d9[1] \n"

          "vmla.f32 %[c0], q15, d10[0] \n"
          "vmla.f32 %[c1], q15, d10[1] \n"
          "vmla.f32 %[c2], q15, d11[0] \n"
          "vmla.f32 %[c3], q15, d11[1] \n"

          "vmla.f32 %[c0], q10, d0[0] \n"
          "vmla.f32 %[c1], q10, d0[1] \n"
          "vmla.f32 %[c2], q10, d1[0] \n"
          "vmla.f32 %[c3], q10, d1[1] \n"

          "subs %[block_d], %[block_d], #1 \n"

          "vmla.f32 %[c0], q11, d2[0] \n"
          "vmla.f32 %[c1], q11, d2[1] \n"
          "vmla.f32 %[c2], q11, d3[0] \n"
          "vmla.f32 %[c3], q11, d3[1] \n"

          "bne 0b \n"
        :  // outputs
          [lhs_ptr] "+r"(lhs_ptr),
          [rhs_ptr] "+r"(rhs_ptr),
          [res_ptr] "+r"(res_ptr),
          [block_d] "+r"(block_d),
          [c0] "+w"(c0),
          [c1] "+w"(c1),
          [c2] "+w"(c2),
          [c3] "+w"(c3)
        :  // inputs
        :  // clabbers
        "cc", "memory",
        "q0", "q1", "q2", "q3", "q4", "q5",
        "q10", "q11", "q12", "q13", "q14", "q15");
      }
#endif  // __aarch64__

      // d: 4
      block_d = remain_d >> 2;
      remain_d -= (block_d << 2);

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

      // d: 8
      block_d = remain_d >> 3;
      remain_d -= (block_d << 3);

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

    const float *lhs_ptr = lhs_data;
    float *res_ptr = result_data + height * bw;

#if defined(MACE_ENABLE_NEON)
    index_t block_h = 0;
#if defined(__aarch64__)
    block_h = remain_h >> 3;
    remain_h -= (block_h << 3);

    // h: 8
    for (index_t bh = 0; bh < block_h; ++bh) {
      const float *rhs_ptr = rhs_data + depth * bw;

      index_t remain_d = depth;

      float32x4_t c0, c1;
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
        float32x4_t a0 = vdupq_n_f32(rhs_ptr[0]);

        b0 = vld1q_f32(lhs_ptr);
        b1 = vld1q_f32(lhs_ptr + 4);

        c0 = vfmaq_laneq_f32(c0, b0, a0, 0);
        c1 = vfmaq_laneq_f32(c1, b1, a0, 0);

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
        float32x2_t a0 = vdup_n_f32(rhs_ptr[0]);

        b0 = vld1q_f32(lhs_ptr);

        c0 = vmlaq_lane_f32(c0, b0, a0, 0);

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

      float sum = 0.f;

#if defined(MACE_ENABLE_NEON)
      index_t block_d = 0;

      float32x4_t c0, c1;
      c0 = vdupq_n_f32(0.f);
      c1 = vdupq_n_f32(0.f);

      block_d = remain_d >> 3;
      remain_d -= (block_d << 3);

      // d: 8
      for (index_t bd = 0; bd < block_d; ++bd) {
        // 1.8.1
        float32x4_t a0, a1;
        float32x4_t b0, b1;

        a0 = vld1q_f32(lhs_ptr);
        a1 = vld1q_f32(lhs_ptr + 4);
        b0 = vld1q_f32(rhs_ptr);
        b1 = vld1q_f32(rhs_ptr + 4);

        c0 = vmlaq_f32(c0, a0, b0);
        c1 = vmlaq_f32(c1, a1, b1);

        lhs_ptr += 8;
        rhs_ptr += 8;
      }

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
      sum += vaddvq_f32(c0);
      sum += vaddvq_f32(c1);
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
                    PackedBlock *packed_block) {
  Pack(lhs, PackOrder::ColMajor, packed_block);
}

void SGemm::PackRhs(const MatrixMap<const float> &rhs,
                    PackedBlock *packed_block) {
  Pack(rhs, PackOrder::RowMajor, packed_block);
}

void SGemm::Pack(const MatrixMap<const float> &src,
                 const PackOrder order,
                 PackedBlock *packed_block) {
  MACE_CHECK_NOTNULL(packed_block);

  const index_t height = src.row();
  const index_t width = src.col();
  auto packed_data = packed_block->mutable_data<float>();

#define MACE_SGEMM_PACK_PER_BATCH                                     \
    for (index_t b = 0; b < src.batch(); ++b) {                       \
      PackPerBatch(src, order, b, packed_data + b * height * width);  \
    }
  if (src.batch() >= MaceOpenMPThreadCount) {
#pragma omp parallel for
    MACE_SGEMM_PACK_PER_BATCH
  } else {
    MACE_SGEMM_PACK_PER_BATCH
  }
#undef MACE_SGEMM_PACK_PER_BATCH
}

void SGemm::UnPack(const PackedBlock &packed_result,
                   MatrixMap<float> *matrix_map) {
  MACE_CHECK_NOTNULL(matrix_map);

  const index_t height = matrix_map->row();
  const index_t width = matrix_map->col();
  auto packed_data = packed_result.data<float>();

#define MACE_SGEMM_UNPACK_PER_BATCH                                   \
  for (index_t b = 0; b < matrix_map->batch(); ++b) {                 \
    UnPackPerBatch(packed_data + b * height * width, b, matrix_map);  \
  }

  if (matrix_map->batch() >= MaceOpenMPThreadCount) {
#pragma omp parallel for
    MACE_SGEMM_UNPACK_PER_BATCH
  } else {
    MACE_SGEMM_UNPACK_PER_BATCH
  }
#undef MACE_SGEMM_UNPACK_PER_BATCH
}

void SGemm::PackPerBatch(const MatrixMap<const float> &src,
                         const PackOrder order,
                         const index_t batch_index,
                         float *packed_data) {
  MACE_CHECK_NOTNULL(packed_data);

  const index_t height = src.row();
  const index_t width = src.col();
  auto src_data = src.batch_data(batch_index);

  if (src.map_major() == Major::RowMajor && order == PackOrder::ColMajor) {
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
  } else if (src.map_major() == Major::ColMajor &&
             order == PackOrder::ColMajor) {
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
  } else if (src.map_major() == Major::RowMajor &&
             order == PackOrder::RowMajor) {
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
  } else if (src.map_major() == Major::ColMajor &&
             order == PackOrder::RowMajor) {
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

void SGemm::UnPackPerBatch(const float *packed_data,
                           const index_t batch_index,
                           MatrixMap<float> *matrix_map) {
  MACE_CHECK_NOTNULL(matrix_map);

  const index_t height = matrix_map->row();
  const index_t width = matrix_map->col();
  auto unpacked_data = matrix_map->batch_data(batch_index);

  if (matrix_map->map_major() == Major::RowMajor) {
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

}  // namespace kernels
}  // namespace mace
