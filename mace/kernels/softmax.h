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

#ifndef MACE_KERNELS_SOFTMAX_H_
#define MACE_KERNELS_SOFTMAX_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <limits>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"
#include "mace/utils/utils.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

template<DeviceType D, typename T>
struct SoftmaxFunctor;

template<>
struct SoftmaxFunctor<DeviceType::CPU, float> {
  void operator()(const Tensor *input, Tensor *output, StatsFuture *future) {
    MACE_UNUSED(future);
    const index_t batch = input->dim(0);
    const index_t class_count = input->dim(1);
    const index_t class_size = input->dim(2) * input->dim(3);
    const index_t batch_size = class_count * class_size;

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();

    for (index_t b = 0; b < batch; ++b) {
#pragma omp parallel for
      for (index_t k = 0; k < class_size; ++k) {
        const float *input_ptr = input_data + b * batch_size + k;
        float *output_ptr = output_data + b * batch_size + k;

        float max_val = std::numeric_limits<float>::lowest();
        index_t channel_offset = 0;
        for (index_t c = 0; c < class_count; ++c) {
          float data = input_ptr[channel_offset];
          if (data > max_val) {
            max_val = data;
          }
          channel_offset += class_size;
        }

        channel_offset = 0;
        float sum = 0;
        for (index_t c = 0; c < class_count; ++c) {
          float exp_value = ::exp(input_ptr[channel_offset] - max_val);
          sum += exp_value;
          output_ptr[channel_offset] = exp_value;
          channel_offset += class_size;
        }

        channel_offset = 0;
        for (index_t c = 0; c < class_count; ++c) {
          output_ptr[channel_offset] /= sum;
          channel_offset += class_size;
        }
      }  // k
    }  // b
  }
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct SoftmaxFunctor<DeviceType::GPU, T> {
  void operator()(const Tensor *logits, Tensor *output, StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_SOFTMAX_H_
