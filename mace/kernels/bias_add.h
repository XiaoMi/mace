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

#include <functional>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/kernel.h"
#include "mace/public/mace.h"

namespace mace {
namespace kernels {

struct BiasAddFunctorBase : OpKernel {
  BiasAddFunctorBase(OpKernelContext *context,
                     const DataFormat data_format)
      : OpKernel(context), data_format_(data_format) {}

  DataFormat data_format_;
};

template <DeviceType D, typename T>
struct BiasAddFunctor;

template <>
struct BiasAddFunctor<DeviceType::CPU, float> : BiasAddFunctorBase {
  BiasAddFunctor(OpKernelContext *context,
                 const DataFormat data_format)
      : BiasAddFunctorBase(context, data_format) {}

  MaceStatus operator()(const Tensor *input,
                        const Tensor *bias,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);

    const float *input_ptr = input->data<float>();
    const float *bias_ptr = bias->data<float>();
    float *output_ptr = output->mutable_data<float>();

    if (input->dim_size() == 4 && data_format_ == NCHW) {
      const index_t batch = input->dim(0);
      const index_t channels = input->dim(1);
      const index_t height_width = input->dim(2) * input->dim(3);

#pragma omp parallel for collapse(2)
      for (index_t n = 0; n < batch; ++n) {
        for (index_t c = 0; c < channels; ++c) {
          for (index_t hw = 0; hw < height_width; ++hw) {
            index_t pos = (n * channels + c) * height_width + hw;
            output_ptr[pos] = input_ptr[pos] + bias_ptr[c];
          }
        }
      }
    } else {
      const std::vector<index_t> &shape = input->shape();
      const index_t fused_batch = std::accumulate(
          shape.begin(), shape.end() - 1, 1, std::multiplies<index_t>());
      const index_t channels = *shape.rbegin();
#pragma omp parallel for
      for (index_t n = 0; n < fused_batch; ++n) {
        index_t pos = n * channels;
        for (index_t c = 0; c < channels; ++c) {
          output_ptr[pos] = input_ptr[pos] + bias_ptr[c];
          ++pos;
        }
      }
    }

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
class OpenCLBiasAddKernel {
 public:
  virtual MaceStatus Compute(
      OpKernelContext *context,
      const Tensor *input,
      const Tensor *bias,
      Tensor *output,
      StatsFuture *future) = 0;
  MACE_VIRTUAL_EMPTY_DESTRUCTOR(OpenCLBiasAddKernel);
};

template <typename T>
struct BiasAddFunctor<DeviceType::GPU, T> : BiasAddFunctorBase {
  BiasAddFunctor(OpKernelContext *context, const DataFormat data_format);
  MaceStatus operator()(const Tensor *input,
                        const Tensor *bias,
                        Tensor *output,
                        StatsFuture *future);

  std::unique_ptr<OpenCLBiasAddKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_BIAS_ADD_H_
