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

#ifndef MACE_KERNELS_BIAS_ADD_H_
#define MACE_KERNELS_BIAS_ADD_H_

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
struct BiasAddFunctor;

template<>
struct BiasAddFunctor<DeviceType::CPU, float> {
  void operator()(const Tensor *input,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    const index_t batch = input->dim(0);
    const index_t channels = input->dim(1);
    const index_t height = input->dim(2);
    const index_t width = input->dim(3);

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);

    const float *input_ptr = input->data<float>();
    const float *bias_ptr = bias->data<float>();
    float *output_ptr = output->mutable_data<float>();

#pragma omp parallel for collapse(2)
    for (index_t n = 0; n < batch; ++n) {
      for (index_t c = 0; c < channels; ++c) {
        for (index_t hw = 0; hw < height * width; ++hw) {
          index_t pos = (n * channels + c) * height * width + hw;
          output_ptr[pos] = input_ptr[pos] + bias_ptr[c];
        }
      }
    }
  }
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct BiasAddFunctor<DeviceType::GPU, T> {
  void operator()(const Tensor *input,
                  const Tensor *bias,
                  Tensor *output,
                  StatsFuture *future);
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_BIAS_ADD_H_
