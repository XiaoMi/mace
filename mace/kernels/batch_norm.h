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

#ifndef MACE_KERNELS_BATCH_NORM_H_
#define MACE_KERNELS_BATCH_NORM_H_

#if defined(MACE_ENABLE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/kernels/activation.h"
#include "mace/public/mace.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

struct BatchNormFunctorBase {
  BatchNormFunctorBase(bool folded_constant,
                       const ActivationType activation,
                       const float relux_max_limit)
    : folded_constant_(folded_constant),
      activation_(activation),
      relux_max_limit_(relux_max_limit) {}

  const bool folded_constant_;
  const ActivationType activation_;
  const float relux_max_limit_;
};

template<DeviceType D, typename T>
struct BatchNormFunctor;

template<>
struct BatchNormFunctor<DeviceType::CPU, float> : BatchNormFunctorBase {
  BatchNormFunctor(const bool folded_constant,
                   const ActivationType activation,
                   const float relux_max_limit)
    : BatchNormFunctorBase(folded_constant, activation, relux_max_limit) {}

  MaceStatus operator()(const Tensor *input,
                  const Tensor *scale,
                  const Tensor *offset,
                  const Tensor *mean,
                  const Tensor *var,
                  const float epsilon,
                  Tensor *output,
                  StatsFuture *future) {
    MACE_UNUSED(future);
    // Batch normalization in the paper https://arxiv.org/abs/1502.03167 .
    // The calculation formula for inference is
    // Y = \frac{ \scale } { \sqrt{var+\variance_epsilon} } * X +
    //          ( \offset - \frac { \scale * mean } {
    //          \sqrt{var+\variance_epsilon} }
    // new_scale = \frac{ \scale } { \sqrt{var+\variance_epsilon} }
    // new_offset = \offset - mean * common_val;
    // Y = new_scale * X + new_offset;
    const index_t batch = input->dim(0);
    const index_t channels = input->dim(1);
    const index_t height = input->dim(2);
    const index_t width = input->dim(3);

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard scale_mapper(scale);
    Tensor::MappingGuard offset_mapper(offset);
    Tensor::MappingGuard output_mapper(output);

    const float *input_ptr = input->data<float>();
    const float *scale_ptr = scale->data<float>();
    const float *offset_ptr = offset->data<float>();
    float *output_ptr = output->mutable_data<float>();

    std::vector<float> new_scale;
    std::vector<float> new_offset;
    if (!folded_constant_) {
      new_scale.resize(channels);
      new_offset.resize(channels);
      Tensor::MappingGuard mean_mapper(mean);
      Tensor::MappingGuard var_mapper(var);
      const float *mean_ptr = mean->data<float>();
      const float *var_ptr = var->data<float>();
#pragma omp parallel for
      for (index_t c = 0; c < channels; ++c) {
        new_scale[c] = scale_ptr[c] / std::sqrt(var_ptr[c] + epsilon);
        new_offset[c] = offset_ptr[c] - mean_ptr[c] * new_scale[c];
      }
    }

    const float *scale_data = folded_constant_ ? scale_ptr : new_scale.data();
    const float
      *offset_data = folded_constant_ ? offset_ptr : new_offset.data();

    index_t channel_size = height * width;
    index_t batch_size = channels * channel_size;

    // NEON is slower, so stick to the trivial implementaion
#pragma omp parallel for collapse(2)
    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channels; ++c) {
        index_t offset = b * batch_size + c * channel_size;
        for (index_t hw = 0; hw < height * width; ++hw) {
          output_ptr[offset + hw] =
            scale_data[c] * input_ptr[offset + hw] + offset_data[c];
        }
      }
    }
    DoActivation(output_ptr, output_ptr, output->size(), activation_,
                 relux_max_limit_);

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct BatchNormFunctor<DeviceType::GPU, T> : BatchNormFunctorBase {
  BatchNormFunctor(const bool folded_constant,
                   const ActivationType activation,
                   const float relux_max_limit)
    : BatchNormFunctorBase(folded_constant, activation, relux_max_limit) {}
  MaceStatus operator()(const Tensor *input,
                  const Tensor *scale,
                  const Tensor *offset,
                  const Tensor *mean,
                  const Tensor *var,
                  const float epsilon,
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

#endif  // MACE_KERNELS_BATCH_NORM_H_
