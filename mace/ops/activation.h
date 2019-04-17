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

#ifndef MACE_OPS_ACTIVATION_H_
#define MACE_OPS_ACTIVATION_H_

#include <algorithm>
#include <cmath>
#include <string>

#include "mace/core/types.h"
#include "mace/core/op_context.h"
#include "mace/ops/common/activation_type.h"
#include "mace/utils/logging.h"

namespace mace {
namespace ops {

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
    return ActivationType::LEAKYRELU;
  } else {
    LOG(FATAL) << "Unknown activation type: " << type;
  }
  return ActivationType::NOOP;
}

template<typename T>
void PReLUActivation(const OpContext *context,
                     const T *input_ptr,
                     const index_t outer_size,
                     const index_t input_chan,
                     const index_t inner_size,
                     const T *alpha_ptr,
                     T *output_ptr) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t i = start0; i < end0; i += step0) {
      for (index_t chan_idx = start1; chan_idx < end1; chan_idx += step1) {
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
  }, 0, outer_size, 1, 0, input_chan, 1);
}

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ACTIVATION_H_
