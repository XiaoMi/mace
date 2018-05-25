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
                        float *out_min_data,
                        float *out_max_data) {
  // re-range to make range include zero float and
  // make zero float as integer u8
  const float quantized_max = std::numeric_limits<uint8_t>::max();
  float out_min = fminf(0.f, in_min_data);
  float out_max = fmaxf(0.f, in_max_data);
  if (out_min < 0.f) {
    float stepsize = (in_max_data - in_min_data) / quantized_max;
    float quantized_zero = -in_min_data / stepsize;
    float quantized_zero_near_int = roundf(quantized_zero);
    if (fabs(quantized_zero - quantized_zero_near_int) > 1e-6) {
      if (quantized_zero < quantized_zero_near_int) {
        // keep out_max fixed, and move out_min
        stepsize = out_max / (quantized_max - quantized_zero_near_int);
        out_min = out_max - quantized_max * stepsize;
      } else {
        // keep out_min fixed, and move out_max
        stepsize = -out_min / quantized_zero_near_int;
        out_max = out_min + quantized_max * stepsize;
      }
    }
  }
  *out_min_data = out_min;
  *out_max_data = out_max;
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

template<DeviceType D, typename T>
struct QuantizeFunctor;

template<>
struct QuantizeFunctor<CPU, uint8_t> {
  QuantizeFunctor() {}

  MaceStatus operator()(const Tensor *input,
                  const Tensor *in_min,
                  const Tensor *in_max,
                  Tensor *output,
                  Tensor *out_min,
                  Tensor *out_max,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    const float *input_data = input->data<float>();
    const float in_min_data = in_min->data<float>()[0];
    const float in_max_data = in_max->data<float>()[0];
    uint8_t *output_data = output->mutable_data<uint8_t>();
    float *out_min_data = out_min->mutable_data<float>();
    float *out_max_data = out_max->mutable_data<float>();

    AdjustRange<uint8_t>(in_min_data, in_max_data, out_min_data, out_max_data);
    float recip_stepsize = 255.f / (out_max_data[0] - out_min_data[0]);
    for (int i = 0; i < input->size(); ++i) {
      output_data[i] = Saturate<uint8_t>(roundf(
        (input_data[i] - in_min_data) * recip_stepsize));
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
                  const Tensor *in_min,
                  const Tensor *in_max,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    const uint8_t *input_data = input->data<uint8_t>();
    const float in_min_data = in_min->data<float>()[0];
    const float in_max_data = in_max->data<float>()[0];
    float *output_data = output->mutable_data<float>();

    float stepsize = (in_max_data - in_min_data) / 255.0;
    for (int i = 0; i < input->size(); ++i) {
      output_data[i] = in_min_data + stepsize * input_data[i];
    }

    return MACE_SUCCESS;
  }
};

template<DeviceType D, typename T>
struct RequantizeFunctor;

template<>
struct RequantizeFunctor<CPU, uint8_t> {
  RequantizeFunctor() {}

  MaceStatus operator()(const Tensor *input,
                  const Tensor *in_min,
                  const Tensor *in_max,
                  const Tensor *rerange_min,
                  const Tensor *rerange_max,
                  Tensor *output,
                  Tensor *out_min,
                  Tensor *out_max,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    const int *input_data = input->data<int>();
    const float in_min_data = in_min->data<float>()[0];
    const float in_max_data = in_max->data<float>()[0];

    float rerange_min_data;
    float rerange_max_data;
    int min_val = std::numeric_limits<int>::max();
    int max_val = std::numeric_limits<int>::lowest();
    double
      si = (in_max_data - in_min_data) / std::numeric_limits<uint32_t>::max();
    if (rerange_min == nullptr && rerange_max == nullptr) {
      for (int i = 0; i < input->size(); ++i) {
        min_val = std::min(min_val, input_data[i]);
        max_val = std::max(max_val, input_data[i]);
      }
      rerange_min_data = min_val * si;
      rerange_max_data = max_val * si;
    } else {
      rerange_min_data = rerange_min->data<float>()[0];
      rerange_max_data = rerange_max->data<float>()[0];
    }

    uint8_t *output_data = output->mutable_data<uint8_t>();
    float *out_min_data = out_min->mutable_data<float>();
    float *out_max_data = out_max->mutable_data<float>();

    AdjustRange<uint8_t>(rerange_min_data,
                         rerange_max_data,
                         out_min_data,
                         out_max_data);
    /**
     * f = qi * si = min_o + qo * so
     * => qo = (qi * si - min_o) / so
     *       = qi * (si/so) - min_o / so
     *       = qi * (si / so) + zo
     *
     *    zo = -min_o / so
     *
     */
    float so =
      (out_max_data[0] - out_min_data[0]) / std::numeric_limits<uint8_t>::max();
    double step_ratio = si / so;
    float quantized_out_zero = -out_min_data[0] / so;

    for (int i = 0; i < output->size(); ++i) {
      output_data[i] =
        Saturate<uint8_t>(roundf(
          quantized_out_zero + input_data[i] * step_ratio));
    }

    return MACE_SUCCESS;
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_QUANTIZE_H_
