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

#ifndef MACE_KERNELS_LOCAL_RESPONSE_NORM_H_
#define MACE_KERNELS_LOCAL_RESPONSE_NORM_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct LocalResponseNormFunctor;

template<>
struct LocalResponseNormFunctor<DeviceType::CPU, float> {
  MaceStatus operator()(const Tensor *input,
                  int depth_radius,
                  float bias,
                  float alpha,
                  float beta,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    const index_t batch = input->dim(0);
    const index_t channels = input->dim(1);
    const index_t height = input->dim(2);
    const index_t width = input->dim(3);

    const float *input_ptr = input->data<float>();
    float *output_ptr = output->mutable_data<float>();

    index_t image_size = height * width;
    index_t batch_size = channels * image_size;

#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channels; ++c) {
        const int begin_input_c = std::max(static_cast<index_t>(0),
                                           c - depth_radius);
        const int end_input_c = std::min(channels, c + depth_radius + 1);

        index_t pos = b * batch_size;
        for (index_t hw = 0; hw < height * width; ++hw, ++pos) {
          float accum = 0.f;
          for (int input_c = begin_input_c; input_c < end_input_c; ++input_c) {
            const float input_val = input_ptr[pos + input_c * image_size];
            accum += input_val * input_val;
          }
          const float multiplier = std::pow(bias + alpha * accum, -beta);
          output_ptr[pos + c * image_size] =
            input_ptr[pos + c * image_size] * multiplier;
        }
      }
    }

    return MACE_SUCCESS;
  }
};

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_LOCAL_RESPONSE_NORM_H_
