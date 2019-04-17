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

#ifndef MACE_CORE_QUANTIZE_H_
#define MACE_CORE_QUANTIZE_H_

#include <algorithm>
#include <cmath>
#include <limits>

#include "mace/utils/logging.h"
#include "mace/utils/thread_pool.h"
#include "mace/core/tensor.h"

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
    if (fabs(quantized_zero - quantized_zero_near_int) > kEps && non_zero) {
      *zero_point = static_cast<int32_t>(std::ceil(quantized_zero));
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

template<typename T>
class QuantizeUtil {
 public:
  explicit QuantizeUtil(utils::ThreadPool *thread_pool)
      : thread_pool_(thread_pool) {}

  void QuantizeWithScaleAndZeropoint(const float *input,
                                     const index_t size,
                                     float scale,
                                     int32_t zero_point,
                                     T *output) {
    float recip_scale = 1 / scale;
    thread_pool_->Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t i = start; i < end; i += step) {
        output[i] = Saturate<T>(roundf(zero_point + recip_scale * input[i]));
      }
    }, 0, size, 1);
  }

  void Quantize(const float *input,
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

  void Quantize(const Tensor &input,
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

  void Dequantize(const T *input,
                  const index_t size,
                  const float scale,
                  const int32_t zero_point,
                  float *output) {
    thread_pool_->Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t i = start; i < end; i += step) {
        output[i] = scale * (input[i] - zero_point);
      }
    }, 0, size, 1);
  }

  void DeQuantize(const Tensor &input,
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

 private:
  utils::ThreadPool *thread_pool_;
};

#ifdef MACE_ENABLE_NEON

template<>
void QuantizeUtil<uint8_t>::QuantizeWithScaleAndZeropoint(
    const float *input,
    const index_t size,
    float scale,
    int32_t zero_point,
    uint8_t *output);

template<>
void QuantizeUtil<uint8_t>::Dequantize(const uint8_t *input,
                                       const index_t size,
                                       const float scale,
                                       const int32_t zero_point,
                                       float *output);

template<>
void QuantizeUtil<int32_t>::Dequantize(const int *input,
                                       const index_t size,
                                       const float scale,
                                       const int32_t zero_point,
                                       float *output);

#endif

}  // namespace mace

#endif  // MACE_CORE_QUANTIZE_H_
