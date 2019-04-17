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

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif  // MACE_ENABLE_NEON

#include "mace/core/quantize.h"

namespace mace {

#ifdef MACE_ENABLE_NEON

template<>
void QuantizeUtil<uint8_t>::QuantizeWithScaleAndZeropoint(
    const float *input,
    const index_t size,
    float scale,
    int32_t zero_point,
    uint8_t *output) {
  const float32x4_t vround = vdupq_n_f32(0.5);
  const float32x4_t
      vzero = vaddq_f32(vround, vcvtq_f32_s32(vdupq_n_s32(zero_point)));
  const float recip_scale = 1.f / scale;
  const float32x4_t vrecip_scale = vdupq_n_f32(recip_scale);
  const index_t block_count = size / 16;

  thread_pool_->Compute1D([=](index_t start, index_t end, index_t step) {
    for (index_t i = start; i < end; i += step) {
      float32x4_t vi0 = vld1q_f32(input + i * 16);
      float32x4_t vi1 = vld1q_f32(input + i * 16 + 4);
      float32x4_t vi2 = vld1q_f32(input + i * 16 + 8);
      float32x4_t vi3 = vld1q_f32(input + i * 16 + 12);

      int32x4_t vo0_s32 = vcvtq_s32_f32(vmlaq_f32(vzero, vi0, vrecip_scale));
      int32x4_t vo1_s32 = vcvtq_s32_f32(vmlaq_f32(vzero, vi1, vrecip_scale));
      int32x4_t vo2_s32 = vcvtq_s32_f32(vmlaq_f32(vzero, vi2, vrecip_scale));
      int32x4_t vo3_s32 = vcvtq_s32_f32(vmlaq_f32(vzero, vi3, vrecip_scale));

      uint8x8_t vo0_u8 =
          vqmovun_s16(vcombine_s16(vqmovn_s32(vo0_s32), vqmovn_s32(vo1_s32)));
      uint8x8_t vo1_u8 =
          vqmovun_s16(vcombine_s16(vqmovn_s32(vo2_s32), vqmovn_s32(vo3_s32)));
      uint8x16_t vo = vcombine_u8(vo0_u8, vo1_u8);

      vst1q_u8(output + i * 16, vo);
    }
  }, 0, block_count, 1);

  for (index_t i = block_count * 16; i < size; ++i) {
    output[i] =
        Saturate<uint8_t>(roundf(zero_point + recip_scale * input[i]));
  }
}

template<>
void QuantizeUtil<uint8_t>::Dequantize(const uint8_t *input,
                                       const index_t size,
                                       const float scale,
                                       const int32_t zero_point,
                                       float *output) {
  const index_t block_count = size / 16;
  const int32x4_t vzero = vdupq_n_s32(zero_point);
  const float32x4_t vscale = vdupq_n_f32(scale);

  thread_pool_->Compute1D([=](index_t start, index_t end, index_t step) {
    for (index_t i = start; i < end; i += step) {
      uint8x16_t vi = vld1q_u8(input + i * 16);
      float32x4x4_t vo = {{
          vmulq_f32(vscale,
                    vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                        vget_low_u16(vmovl_u8(vget_low_u8(vi))))), vzero))),
          vmulq_f32(vscale,
                    vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                        vget_high_u16(vmovl_u8(vget_low_u8(vi))))), vzero))),
          vmulq_f32(vscale,
                    vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                        vget_low_u16(vmovl_u8(vget_high_u8(vi))))), vzero))),
          vmulq_f32(vscale,
                    vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(
                        vget_high_u16(vmovl_u8(vget_high_u8(vi))))), vzero))),
      }};
      vst1q_f32(output + i * 16, vo.val[0]);
      vst1q_f32(output + i * 16 + 4, vo.val[1]);
      vst1q_f32(output + i * 16 + 8, vo.val[2]);
      vst1q_f32(output + i * 16 + 12, vo.val[3]);
    }
  }, 0, block_count, 1);

  for (index_t i = block_count * 16; i < size; ++i) {
    output[i] = scale * (input[i] - zero_point);
  }
}

template<>
void QuantizeUtil<int32_t>::Dequantize(const int *input,
                                       const index_t size,
                                       const float scale,
                                       const int32_t zero_point,
                                       float *output) {
  const index_t block_count = size / 4;
  const int32x4_t vzero = vdupq_n_s32(zero_point);
  const float32x4_t vscale = vdupq_n_f32(scale);

  thread_pool_->Compute1D([=](index_t start, index_t end, index_t step) {
    for (index_t i = start; i < end; i += step) {
      int32x4_t vi = vld1q_s32(input + i * 4);
      float32x4_t vo = vmulq_f32(vscale, vcvtq_f32_s32(vsubq_s32(vi, vzero)));
      vst1q_f32(output + i * 4, vo);
    }
  }, 0, block_count, 1);

  for (index_t i = block_count * 4; i < size; ++i) {
    output[i] = scale * (input[i] - zero_point);
  }
}
#endif

}  // namespace mace
