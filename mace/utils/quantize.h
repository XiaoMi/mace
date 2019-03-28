// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_UTILS_QUANTIZE_H_
#define MACE_UTILS_QUANTIZE_H_

#include <algorithm>
#include <cmath>
#include <limits>

#if defined(MACE_ENABLE_NEON)
#include <arm_neon.h>
#endif  // MACE_ENABLE_NEON

#include "mace/utils/logging.h"

namespace mace {

template<typename T>
inline void AdjustRange(const float in_min_data,
                        const float in_max_data,
                        const bool non_zero,
                        float *scale,
                        int32_t *zero_point) {
  // re-range to make range include zero float and
  // make zero float as integer u8
  const T quantized_min = std::numeric_limits<T>::lowest();
  const T quantized_max = std::numeric_limits<T>::max();
  if (quantized_min < 0) {
    MACE_ASSERT(!non_zero, "Cannot nudge to non_zero quantize value.");
  }

  float out_max = std::max(0.f, in_max_data);
  float out_min = std::min(0.f, in_min_data);
  // make in_min_data quantize as greater than 1
  if (non_zero) {
    out_min = std::min(out_min,
                       in_min_data - (out_max - in_min_data)
                           / (quantized_max - quantized_min - 1));
  }

  *scale = (out_max - out_min) / (quantized_max - quantized_min);
  const float kEps = 1e-6;
  if (out_min < -kEps && out_max > kEps) {
    float quantized_zero = -out_min / *scale;
    int32_t
        quantized_zero_near_int = static_cast<int32_t>(roundf(quantized_zero));
    *zero_point = quantized_zero_near_int;
    if (fabs(quantized_zero - quantized_zero_near_int) > kEps) {
      if (quantized_zero < quantized_zero_near_int || non_zero) {
        // keep out_max fixed, and move out_min
        *zero_point = static_cast<int32_t>(std::ceil(quantized_zero));
        *scale = out_max / (quantized_max - *zero_point);
      } else {
        // keep out_min fixed, and move out_max
        *scale = out_min / (quantized_min - *zero_point);
      }
    }
  } else if (out_min > -kEps) {
    *zero_point = quantized_min;
  } else {
    *zero_point = quantized_max;
  }
}

template<typename T>
inline T Saturate(float value) {
  int rounded_value = static_cast<int>(value);
  if (rounded_value <= std::numeric_limits<T>::lowest()) {
    return std::numeric_limits<T>::lowest();
  } else if (rounded_value >= std::numeric_limits<T>::max()) {
    return std::numeric_limits<T>::max();
  } else {
    return static_cast<T>(rounded_value);
  }
}

inline void FindMinMax(const float *input,
                       const index_t size,
                       float *min_val, float *max_val) {
  float max_v = std::numeric_limits<float>::lowest();
  float min_v = std::numeric_limits<float>::max();
  for (index_t i = 0; i < size; ++i) {
    max_v = std::max(max_v, input[i]);
    min_v = std::min(min_v, input[i]);
  }
  *min_val = min_v;
  *max_val = max_v;
}

template<typename T>
inline void QuantizeWithScaleAndZeropoint(const float *input,
                                          const index_t size,
                                          float scale,
                                          int32_t zero_point,
                                          T *output) {
  float recip_scale = 1 / scale;
#pragma omp parallel for schedule(runtime)
  for (int i = 0; i < size; ++i) {
    output[i] = Saturate<T>(roundf(zero_point + recip_scale * input[i]));
  }
}

template<typename T>
inline void Quantize(const float *input,
                     const index_t size,
                     bool non_zero,
                     T *output,
                     float *scale,
                     int32_t *zero_point) {
  float in_min_data;
  float in_max_data;
  FindMinMax(input, size, &in_min_data, &in_max_data);

  AdjustRange<T>(in_min_data, in_max_data, non_zero,
                 scale, zero_point);

  QuantizeWithScaleAndZeropoint(input, size, *scale, *zero_point, output);
}

template<typename T>
inline void Quantize(const Tensor &input,
                     Tensor *output,
                     float *min_out,
                     float *max_out) {
  MACE_CHECK(input.size() != 0);
  Tensor::MappingGuard input_guard(&input);
  Tensor::MappingGuard output_guard(output);
  auto *input_data = input.data<float>();
  auto *output_data = output->mutable_data<T>();
  float scale;
  int32_t zero_point;

  Quantize(input_data, input.size(), false, output_data, &scale, &zero_point);

  *min_out = scale * (std::numeric_limits<T>::lowest() - zero_point);
  *max_out = scale * (std::numeric_limits<T>::max() - zero_point);
}

template<typename T>
inline void Dequantize(const T *input,
                       const index_t size,
                       const float scale,
                       const int32_t zero_point,
                       float *output) {
#pragma omp parallel for schedule(runtime)
  for (int i = 0; i < size; ++i) {
    output[i] = scale * (input[i] - zero_point);
  }
}

#if defined(MACE_ENABLE_NEON)
template<>
inline void QuantizeWithScaleAndZeropoint<uint8_t>(const float *input,
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

#pragma omp parallel for schedule(runtime)
  for (index_t i = 0; i < block_count; ++i) {
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

#pragma omp parallel for schedule(runtime)
  for (index_t i = block_count * 16; i < size; ++i) {
    output[i] = Saturate<uint8_t>(roundf(zero_point + recip_scale * input[i]));
  }
}

template<>
inline void Dequantize<int32_t>(const int32_t *input,
                                const index_t size,
                                const float scale,
                                const int32_t zero_point,
                                float *output) {
  const index_t block_count = size / 4;
  const int32x4_t vzero = vdupq_n_s32(zero_point);
  const float32x4_t vscale = vdupq_n_f32(scale);

#pragma omp parallel for schedule(runtime)
  for (index_t i = 0; i < block_count; ++i) {
    int32x4_t vi = vld1q_s32(input + i * 4);
    float32x4_t vo = vmulq_f32(vscale, vcvtq_f32_s32(vsubq_s32(vi, vzero)));
    vst1q_f32(output + i * 4, vo);
  }
  for (index_t i = block_count * 4; i < size; ++i) {
    output[i] = scale * (input[i] - zero_point);
  }
}

template<>
inline void Dequantize<uint8_t>(const uint8_t *input,
                                const index_t size,
                                const float scale,
                                const int32_t zero_point,
                                float *output) {
  const index_t block_count = size / 16;
  const int32x4_t vzero = vdupq_n_s32(zero_point);
  const float32x4_t vscale = vdupq_n_f32(scale);

#pragma omp parallel for schedule(runtime)
  for (index_t i = 0; i < block_count; ++i) {
    uint8x16_t vi = vld1q_u8(input + i * 16);
    float32x4x4_t vo = {
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
    };
    vst1q_f32(output + i * 16, vo.val[0]);
    vst1q_f32(output + i * 16 + 4, vo.val[1]);
    vst1q_f32(output + i * 16 + 8, vo.val[2]);
    vst1q_f32(output + i * 16 + 12, vo.val[3]);
  }
  for (index_t i = block_count * 16; i < size; ++i) {
    output[i] = scale * (input[i] - zero_point);
  }
}
#endif  // MACE_ENABLE_NEON

template<typename T>
inline void DeQuantize(const Tensor &input,
                       const float min_in,
                       const float max_in,
                       Tensor *output) {
  MACE_CHECK(input.size() != 0);
  Tensor::MappingGuard input_guard(&input);
  Tensor::MappingGuard output_guard(output);
  auto *input_data = input.data<T>();
  auto *output_data = output->mutable_data<float>();
  float scale;
  int32_t zero_point;

  AdjustRange<T>(min_in, max_in, false, &scale, &zero_point);

  Dequantize(input_data, input.size(), scale, zero_point, output_data);
}

inline void QuantizeMultiplier(double multiplier,
                               int32_t *output_multiplier,
                               int32_t *shift) {
  const double q = std::frexp(multiplier, shift);
  auto qint = static_cast<int64_t>(roundl(q * (1ll << 31)));
  if (qint == (1ll << 31)) {
    qint /= 2;
    ++*shift;
  }
  *output_multiplier = static_cast<int32_t>(qint);
  MACE_CHECK(*output_multiplier <= std::numeric_limits<int32_t>::max());
}

inline void GetOutputMultiplierAndShift(
    const float lhs_scale, const float rhs_scale, const float output_scale,
    int32_t *quantized_multiplier, int *right_shift) {
  float real_multiplier = lhs_scale * rhs_scale / output_scale;
  MACE_CHECK(real_multiplier > 0.f && real_multiplier < 1.f, real_multiplier);

  int exponent;
  QuantizeMultiplier(real_multiplier, quantized_multiplier, &exponent);
  *right_shift = -exponent;
  MACE_CHECK(*right_shift >= 0);
}

}  // namespace mace

#endif  // MACE_UTILS_QUANTIZE_H_
