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

#ifndef MACE_OPS_ARM_FIXPOINT_GEMM_H_
#define MACE_OPS_ARM_FIXPOINT_GEMM_H_

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif

#if defined(MACE_ENABLE_NEON) && !defined(__aarch64__)
#define vaddvq_u32(v) ((v)[0] + (v)[1] + (v)[2] + (v)[3])
#endif

namespace mace {
namespace ops {

template<typename INPUT_TYPE, typename OUTPUT_TYPE>
void FixPointGemv(const INPUT_TYPE *lhs,
                  const INPUT_TYPE *rhs,
                  const int lhs_zero_point,
                  const int rhs_zero_point,
                  const index_t lhs_height,
                  const index_t lhs_width,
                  OUTPUT_TYPE *result);

template<>
void FixPointGemv<uint8_t, int32_t>(const uint8_t *lhs,
                                    const uint8_t *rhs,
                                    const int lhs_zero_point,
                                    const int rhs_zero_point,
                                    const index_t lhs_height,
                                    const index_t lhs_width,
                                    int32_t *result) {
  int32_t zero_point_dot = lhs_zero_point * rhs_zero_point * lhs_width;

  uint32_t sum_rhs = 0;
  for (index_t i = 0; i < lhs_width; ++i) {
    sum_rhs += rhs[i];
  }

#pragma omp parallel for
  for (index_t h = 0; h < lhs_height; ++h) {
    const uint8_t *lhs_ptr = lhs + h * lhs_width;
    const uint8_t *rhs_ptr = rhs;
    int32_t *ret_ptr = result + h;

    uint32_t dot = 0;
    uint32_t sum_lhs = 0;
    index_t w = 0;

#if defined(MACE_ENABLE_NEON)
    uint32x4_t vo0_high_u32, vo0_low_u32, vo1_high_u32, vo1_low_u32;
    vo0_high_u32 = vdupq_n_u32(0);
    vo0_low_u32 = vdupq_n_u32(0);
    vo1_high_u32 = vdupq_n_u32(0);
    vo1_low_u32 = vdupq_n_u32(0);

    uint32x4_t sum_lhs_low_u32, sum_lhs_high_u32;
    sum_lhs_low_u32 = vdupq_n_u32(0);
    sum_lhs_high_u32 = vdupq_n_u32(0);

    for (; w <= lhs_width - 16; w += 16) {
      uint8x8_t vl0_u8, vl1_u8;
      uint8x8_t vr0_u8, vr1_u8;
      uint16x8_t vl0_u16, vl1_u16;
      uint16x8_t vr0_u16, vr1_u16;

      vl0_u8 = vld1_u8(lhs_ptr);
      vl1_u8 = vld1_u8(lhs_ptr + 8);

      vr0_u8 = vld1_u8(rhs_ptr);
      vr1_u8 = vld1_u8(rhs_ptr + 8);

      vl0_u16 = vmovl_u8(vl0_u8);
      vl1_u16 = vmovl_u8(vl1_u8);

      vr0_u16 = vmovl_u8(vr0_u8);
      vr1_u16 = vmovl_u8(vr1_u8);

      vo0_high_u32 = vmlal_u16(vo0_high_u32,
                               vget_high_u16(vl0_u16),
                               vget_high_u16(vr0_u16));
      vo0_low_u32 = vmlal_u16(vo0_low_u32,
                              vget_low_u16(vl0_u16),
                              vget_low_u16(vr0_u16));
      vo1_high_u32 = vmlal_u16(vo1_high_u32,
                               vget_high_u16(vl1_u16),
                               vget_high_u16(vr1_u16));
      vo1_low_u32 = vmlal_u16(vo1_low_u32,
                              vget_low_u16(vl1_u16),
                              vget_low_u16(vr1_u16));

      // It can be precuculated if lhs is const, but for this case
      // computation is not bottleneck
      sum_lhs_high_u32 += vaddl_u16(vget_high_u16(vl0_u16),
                                   vget_high_u16(vl1_u16));
      sum_lhs_low_u32 += vaddl_u16(vget_low_u16(vl0_u16),
                                  vget_low_u16(vl1_u16));

      lhs_ptr += 16;
      rhs_ptr += 16;
    }
    vo0_low_u32 = vaddq_u32(vo0_high_u32, vo0_low_u32);
    vo1_low_u32 = vaddq_u32(vo1_high_u32, vo1_low_u32);
    vo0_low_u32 = vaddq_u32(vo0_low_u32, vo1_low_u32);
    dot += vaddvq_u32(vo0_low_u32);

    sum_lhs_low_u32 = vaddq_u32(sum_lhs_high_u32, sum_lhs_low_u32);
    sum_lhs = vaddvq_u32(sum_lhs_low_u32);
#endif  // MACE_ENABLE_NEON

    for (; w < lhs_width; ++w) {
      dot += (*lhs_ptr) * (*rhs_ptr);
      sum_lhs += (*lhs_ptr);
      ++lhs_ptr;
      ++rhs_ptr;
    }

    int32_t ret = dot - sum_lhs * rhs_zero_point - sum_rhs * lhs_zero_point
        + zero_point_dot;

    *ret_ptr = ret;
  }  // h
}

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FIXPOINT_GEMM_H_
