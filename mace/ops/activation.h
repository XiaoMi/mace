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

#ifndef MACE_OPS_ACTIVATION_H_
#define MACE_OPS_ACTIVATION_H_

#include <algorithm>
#include <cmath>
#include <string>

#include "mace/core/types.h"
#include "mace/ops/arm/activation_neon.h"
#include "mace/utils/logging.h"

namespace mace {
namespace ops {

enum ActivationType {
  NOOP = 0,
  RELU = 1,
  RELUX = 2,
  PRELU = 3,
  TANH = 4,
  SIGMOID = 5,
  LEAKYRELU = 6,
};

inline ActivationType StringToActivationType(const std::string type) {
  if (type == "RELU") {
    return ActivationType::RELU;
  } else if (type == "RELUX") {
    return ActivationType::RELUX;
  } else if (type == "PRELU") {
    return ActivationType::PRELU;
  } else if (type == "TANH") {
    return ActivationType::TANH;
  } else if (type == "SIGMOID") {
    return ActivationType::SIGMOID;
  } else if (type == "NOOP") {
    return ActivationType::NOOP;
  } else if (type == "LEAKYRELU") {
    return ActivationType ::LEAKYRELU;
  } else {
    LOG(FATAL) << "Unknown activation type: " << type;
  }
  return ActivationType::NOOP;
}

template <typename T>
void DoActivation(const T *input_ptr,
                  T *output_ptr,
                  const index_t size,
                  const ActivationType type,
                  const float relux_max_limit,
                  const float leakyrelu_coefficient) {
  MACE_CHECK(DataTypeToEnum<T>::value != DataType::DT_HALF);

  switch (type) {
    case NOOP:
      break;
    case RELU:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output_ptr[i] = std::max(input_ptr[i], static_cast<T>(0));
      }
      break;
    case RELUX:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output_ptr[i] = std::min(std::max(input_ptr[i], static_cast<T>(0)),
                                 static_cast<T>(relux_max_limit));
      }
      break;
    case TANH:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output_ptr[i] = std::tanh(input_ptr[i]);
      }
      break;
    case SIGMOID:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output_ptr[i] = 1 / (1 + std::exp(-input_ptr[i]));
      }
      break;
    case LEAKYRELU:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output_ptr[i] = std::max(input_ptr[i], static_cast<T>(0))
          + leakyrelu_coefficient * std::min(input_ptr[i], static_cast<T>(0));
      }
      break;
    default:
      LOG(FATAL) << "Unknown activation type: " << type;
  }
}

template<>
inline void DoActivation(const float *input_ptr,
                         float *output_ptr,
                         const index_t size,
                         const ActivationType type,
                         const float relux_max_limit,
                         const float leakyrelu_coefficient) {
  switch (type) {
    case NOOP:
      break;
    case RELU:
      ReluNeon(input_ptr, size, output_ptr);
      break;
    case RELUX:
      ReluxNeon(input_ptr, relux_max_limit, size, output_ptr);
      break;
    case TANH:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output_ptr[i] = std::tanh(input_ptr[i]);
      }
      break;
    case SIGMOID:
#pragma omp parallel for schedule(runtime)
      for (index_t i = 0; i < size; ++i) {
        output_ptr[i] = 1 / (1 + std::exp(-input_ptr[i]));
      }
      break;
    case LEAKYRELU:
      LeakyReluNeon(input_ptr, leakyrelu_coefficient, size, output_ptr);
      break;
    default:
      LOG(FATAL) << "Unknown activation type: " << type;
  }
}

template <typename T>
void PReLUActivation(const T *input_ptr,
                     const index_t outer_size,
                     const index_t input_chan,
                     const index_t inner_size,
                     const T *alpha_ptr,
                     T *output_ptr) {
#pragma omp parallel for collapse(3) schedule(runtime)
  for (index_t i = 0; i < outer_size; ++i) {
    for (index_t chan_idx = 0; chan_idx < input_chan; ++chan_idx) {
      for (index_t j = 0; j < inner_size; ++j) {
        index_t idx = i * input_chan * inner_size + chan_idx * inner_size + j;
        if (input_ptr[idx] < 0) {
          output_ptr[idx] = input_ptr[idx] * alpha_ptr[chan_idx];
        } else {
          output_ptr[idx] = input_ptr[idx];
        }
      }
    }
  }
}

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ACTIVATION_H_
