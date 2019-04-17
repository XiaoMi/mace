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


#include "mace/ops/arm/q8/gemv.h"

#include <arm_neon.h>
#include <algorithm>

#include "mace/utils/math.h"
#include "mace/core/quantize.h"

#if !defined(__aarch64__)

#define vaddvq_u32(v) ((v)[0] + (v)[1] + (v)[2] + (v)[3])

#endif

namespace mace {
namespace ops {
namespace arm {
namespace q8 {

template<typename OUTPUT_TYPE>
MaceStatus Gemv<OUTPUT_TYPE>::Compute(const OpContext *context,
                                      const Tensor *lhs,
                                      const Tensor *rhs,
                                      const Tensor *bias,
                                      const index_t batch,
                                      const index_t lhs_height,
                                      const index_t lhs_width,
                                      const bool lhs_batched,
                                      const bool rhs_batched,
                                      Tensor *output) {
  MACE_UNUSED(context);

  Tensor::MappingGuard lhs_guard(lhs);
  Tensor::MappingGuard rhs_guard(rhs);
  Tensor::MappingGuard bias_guard(bias);
  Tensor::MappingGuard output_guard(output);

  const auto *lhs_data = lhs->data<uint8_t>();
  const auto *rhs_data = rhs->data<uint8_t>();
  OUTPUT_TYPE *output_data = output->mutable_data<OUTPUT_TYPE>();

  float output_multiplier_float = 0.0;
  int32_t output_multiplier = 0;
  int32_t output_shift = 0;
  if (is_output_type_uint8_) {
    MACE_CHECK(output->scale() > 0, "output scale must not be zero");
    output_multiplier_float = lhs->scale() * rhs->scale() / output->scale();
    GetOutputMultiplierAndShift(lhs->scale(),
                                rhs->scale(),
                                output->scale(),
                                &output_multiplier,
                                &output_shift);
  }

  const int32_t lhs_zero_point = lhs->zero_point();
  const int32_t rhs_zero_point = rhs->zero_point();

  const index_t w_block_size = 16;
  const index_t w_block_count = lhs_width / w_block_size;
  const index_t w_block_remain = lhs_width - w_block_size * w_block_count;

  for (index_t b = 0; b < batch; ++b) {
    const uint8_t *rhs_base =
        rhs_data + static_cast<index_t>(rhs_batched) * b * lhs_width;
    uint32_t sum_rhs = 0;
    for (index_t i = 0; i < lhs_width; ++i) {
      sum_rhs += static_cast<uint32_t>(rhs_base[i]);
    }

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t h = start; h < end; h += step) {
        const uint8_t *lhs_ptr = lhs_data
            + static_cast<index_t>(lhs_batched) * b * lhs_height * lhs_width
            + h * lhs_width;
        const uint8_t *rhs_ptr = rhs_base;
        OUTPUT_TYPE *output_ptr = output_data + b * lhs_height + h;

        uint32_t dot = 0;
        uint32_t sum_lhs = 0;
        uint32x4_t vo0_high_u32 = vdupq_n_u32(0);
        uint32x4_t vo0_low_u32 = vdupq_n_u32(0);
        uint32x4_t vo1_high_u32 = vdupq_n_u32(0);
        uint32x4_t vo1_low_u32 = vdupq_n_u32(0);
        uint32x4_t sum_lhs_low_u32 = vdupq_n_u32(0);
        uint32x4_t sum_lhs_high_u32 = vdupq_n_u32(0);

        for (index_t w_block_idx = 0; w_block_idx < w_block_count;
             ++w_block_idx) {
          uint8x8_t vl0_u8 = vld1_u8(lhs_ptr);
          uint8x8_t vl1_u8 = vld1_u8(lhs_ptr + 8);

          uint8x8_t vr0_u8 = vld1_u8(rhs_ptr);
          uint8x8_t vr1_u8 = vld1_u8(rhs_ptr + 8);

          uint16x8_t vl0_u16 = vmovl_u8(vl0_u8);
          uint16x8_t vl1_u16 = vmovl_u8(vl1_u8);

          uint16x8_t vr0_u16 = vmovl_u8(vr0_u8);
          uint16x8_t vr1_u16 = vmovl_u8(vr1_u8);

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

          // It can be precalculated if lhs is const, but for this case
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

        for (index_t w = 0; w < w_block_remain; ++w) {
          dot += (*lhs_ptr) * (*rhs_ptr);
          sum_lhs += (*lhs_ptr);
          ++lhs_ptr;
          ++rhs_ptr;
        }

        const auto zero_point_dot =
            static_cast<int32_t>(lhs_zero_point * rhs_zero_point * lhs_width);
        int32_t ret = dot - sum_lhs * rhs_zero_point - sum_rhs * lhs_zero_point
            + zero_point_dot;
        if (bias) {
          ret += bias->data<int32_t>()[h];
        }

        if (is_output_type_uint8_) {
          *output_ptr =
              Saturate<uint8_t>(std::roundf(ret * output_multiplier_float));
        } else {
          *output_ptr = ret;
        }
      }  // h
    }, 0, lhs_height, 1);
  }  // b


  return MaceStatus::MACE_SUCCESS;
}

template
class Gemv<uint8_t>;
template
class Gemv<int32_t>;

}  // namespace q8
}  // namespace arm
}  // namespace ops
}  // namespace mace

#ifdef vaddvq_u32
#undef vaddvq_u32
#endif  // vaddvq_u32
