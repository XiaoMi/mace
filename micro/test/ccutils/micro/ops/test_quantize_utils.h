// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MICRO_TEST_CCUTILS_MICRO_OPS_TEST_QUANTIZE_UTILS_H_
#define MICRO_TEST_CCUTILS_MICRO_OPS_TEST_QUANTIZE_UTILS_H_

#include <math.h>
#include <stdint.h>

#include <limits>

#include "micro/base/logging.h"
#include "micro/common/global_buffer.h"
#include "micro/include/public/micro.h"
#include "micro/port/api.h"

namespace micro {
namespace ops {
namespace test {

template <typename Q>
inline Q Saturate(float value) {
  int rounded_value = static_cast<int>(value);
  if (rounded_value <= std::numeric_limits<Q>::lowest()) {
    return std::numeric_limits<Q>::lowest();
  } else if (rounded_value >= std::numeric_limits<Q>::max()) {
    return std::numeric_limits<Q>::max();
  } else {
    return static_cast<Q>(rounded_value);
  }
}

inline void FindMinMax(const float *input,
                       const uint32_t size,
                       float *min_val,
                       float *max_val) {
  float max_v = base::lowest();
  float min_v = base::highest();
  for (uint32_t i = 0; i < size; ++i) {
    max_v = base::max(max_v, input[i]);
    min_v = base::min(min_v, input[i]);
  }
  *min_val = min_v;
  *max_val = max_v;
}

template <typename Q>
inline void QuantizeWithScaleAndZeropoint(const float *input,
                                          const uint32_t size,
                                          float scale,
                                          int32_t zero_point,
                                          Q *output) {
  float recip_scale = 1 / scale;
  for (uint32_t i = 0; i < size; ++i) {
    output[i] = Saturate<Q>(roundf(zero_point + recip_scale * input[i]));
  }
}

inline void AdjustRangeInt8(const float *input,
                            const uint32_t size,
                            float *scale,
                            int32_t *zero_point) {
  float in_min_data;
  float in_max_data;
  FindMinMax(input, size, &in_min_data, &in_max_data);
  in_max_data = base::max(0.f, in_max_data);
  in_min_data = base::min(0.f, in_min_data);

  *scale = (in_max_data - in_min_data) / 255;
  *zero_point = int8_t(-in_min_data / *scale - 128);
}

inline void AdjustRangeInt8Symmetric(const float *input,
                                     const uint32_t size,
                                     float *scale) {
  float in_min_data;
  float in_max_data;
  FindMinMax(input, size, &in_min_data, &in_max_data);
  in_max_data = base::max(0.f, in_max_data);
  in_min_data = base::min(0.f, in_min_data);

  float max_abs = base::max(base::abs(in_max_data), base::abs(in_min_data));

  *scale = max_abs / 127.0f;
}

inline void AutoQuantizeInt8(const float *input,
                             const uint32_t size,
                             int8_t *output,
                             float *scale,
                             int32_t *zero_point) {
  AdjustRangeInt8(input, size, scale, zero_point);
  QuantizeWithScaleAndZeropoint(input, size, *scale, *zero_point, output);
}

inline void AutoQuantizeInt8Symmetric(const float *input,
                                      const uint32_t size,
                                      int8_t *output,
                                      float *scale) {
  AdjustRangeInt8Symmetric(input, size, scale);
  QuantizeWithScaleAndZeropoint(input, size, *scale, 0, output);
}

inline void Dequantize(const int8_t *input,
                       const uint32_t size,
                       const float scale,
                       const int32_t zero_point,
                       float *output) {
  for (uint32_t i = 0; i < size; ++i) {
    output[i] = static_cast<float>(scale * (input[i] - zero_point));
  }
}

}  // namespace test
}  // namespace ops
}  // namespace micro

#endif  // MICRO_TEST_CCUTILS_MICRO_OPS_TEST_QUANTIZE_UTILS_H_
