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

#ifndef MACE_KERNELS_QUANTIZE_H_
#define MACE_KERNELS_QUANTIZE_H_

#include <vector>
#include <algorithm>
#include <limits>

#include "mace/core/future.h"
#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

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
inline void Dequantize(const T *input,
                       const index_t size,
                       const float scale,
                       const int32_t zero_point,
                       float *output) {
  for (int i = 0; i < size; ++i) {
    output[i] = scale * (input[i] - zero_point);
  }
}

inline void QuantizeMultiplier(double multiplier,
                               int32_t* output_multiplier,
                               int32_t* shift) {
  if (multiplier == 0.f) {
    *output_multiplier = 0;
    *shift = 0;
    return;
  }
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

template<DeviceType D, typename T>
struct QuantizeFunctor;

template<>
struct QuantizeFunctor<CPU, uint8_t> {
  QuantizeFunctor() {}

  MaceStatus operator()(const Tensor *input,
                        const bool non_zero,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const float *input_data = input->data<float>();
    uint8_t *output_data = output->mutable_data<uint8_t>();
    if (output->scale() > 0.f) {
      QuantizeWithScaleAndZeropoint(input_data,
                                    input->size(),
                                    output->scale(),
                                    output->zero_point(),
                                    output_data);
    } else {
      float scale;
      int32_t zero_point;
      Quantize(input_data,
               input->size(),
               non_zero,
               output_data,
               &scale,
               &zero_point);
      output->SetScale(scale);
      output->SetZeroPoint(zero_point);
    }

    return MACE_SUCCESS;
  }
};

template<DeviceType D, typename T>
struct DequantizeFunctor;

template<>
struct DequantizeFunctor<CPU, uint8_t> {
  DequantizeFunctor() {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const uint8_t *input_data = input->data<uint8_t>();
    float *output_data = output->mutable_data<float>();
    Dequantize(input_data,
               input->size(),
               input->scale(),
               input->zero_point(),
               output_data);

    return MACE_SUCCESS;
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_QUANTIZE_H_
