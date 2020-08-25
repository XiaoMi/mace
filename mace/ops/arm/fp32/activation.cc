// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#include <arm_neon.h>
#include <algorithm>

#include "mace/ops/arm/base/activation.h"

namespace mace {
namespace ops {
namespace arm {

template<>
void Activation<float>::ActivateRelu(utils::ThreadPool *thread_pool,
                                     const Tensor *input,
                                     Tensor *output) {
  const auto input_data = input->data<float>();
  auto output_data = output->mutable_data<float>();
  const index_t input_size = input->size();
  const float32x4_t vzero = vdupq_n_f32(0.f);
  const index_t block_count = input_size / 4;

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        auto input_ptr = input_data + start * 4;
        auto output_ptr = output_data + start * 4;

        for (index_t i = start; i < end; i += step) {
          float32x4_t v = vld1q_f32(input_ptr);
          v = vmaxq_f32(v, vzero);
          vst1q_f32(output_ptr, v);

          input_ptr += 4;
          output_ptr += 4;
        }
      },
      0, block_count, 1);

  // remain
  for (index_t i = block_count * 4; i < input_size; ++i) {
    output_data[i] = std::max(0.f, input_data[i]);
  }
}

template<>
void Activation<float>::ActivateRelux(utils::ThreadPool *thread_pool,
                                      const Tensor *input,
                                      Tensor *output) {
  const auto input_data = input->data<float>();
  auto output_data = output->mutable_data<float>();
  const index_t input_size = input->size();
  const float32x4_t vzero = vdupq_n_f32(0.f);
  const float32x4_t vlimit = vdupq_n_f32(limit_);
  const index_t block_count = input_size / 4;

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        auto input_ptr = input_data + start * 4;
        auto output_ptr = output_data + start * 4;

        for (index_t i = start; i < end; i += step) {
          float32x4_t v = vld1q_f32(input_ptr);
          v = vmaxq_f32(v, vzero);
          v = vminq_f32(v, vlimit);
          vst1q_f32(output_ptr, v);

          input_ptr += 4;
          output_ptr += 4;
        }
      },
      0, block_count, 1);

  // remain
  for (index_t i = block_count * 4; i < input_size; ++i) {
    output_data[i] = std::max(0.f, std::min(limit_, input_data[i]));
  }
}

template<>
void Activation<float>::ActivateLeakyRelu(utils::ThreadPool *thread_pool,
                                          const Tensor *input,
                                          Tensor *output) {
  const auto input_data = input->data<float>();
  auto output_data = output->mutable_data<float>();
  const index_t input_size = input->size();
  const float32x4_t vzero = vdupq_n_f32(0.f);
  const float32x4_t valpha = vdupq_n_f32(leakyrelu_coefficient_);
  const index_t block_count = input_size / 4;

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        auto input_ptr = input_data + start * 4;
        auto output_ptr = output_data + start * 4;

        for (index_t i = start; i < end; i += step) {
          float32x4_t v = vld1q_f32(input_ptr);
          float32x4_t u = vminq_f32(v, vzero);
          v = vmaxq_f32(v, vzero);
          v = vmlaq_f32(v, valpha, u);
          vst1q_f32(output_ptr, v);

          input_ptr += 4;
          output_ptr += 4;
        }
      },
      0, block_count, 1);

  // remain
  for (index_t i = block_count * 4; i < input_size; ++i) {
    output_data[i] = std::max(input_data[i], 0.f) +
        std::min(input_data[i], 0.f) * leakyrelu_coefficient_;
  }
}

template<>
void Activation<float>::ActivateTanh(utils::ThreadPool *thread_pool,
                                     const Tensor *input,
                                     Tensor *output) {
  const auto input_data = input->data<float>();
  auto output_data = output->mutable_data<float>();
  const index_t input_size = input->size();

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          output_data[i] = std::tanh(input_data[i]);
        }
      },
      0, input_size, 1);
}

template<>
void Activation<float>::ActivateSigmoid(utils::ThreadPool *thread_pool,
                                        const Tensor *input,
                                        Tensor *output) {
  const auto input_data = input->data<float>();
  auto output_data = output->mutable_data<float>();
  const index_t input_size = input->size();

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        for (index_t i = start; i < end; i += step) {
          output_data[i] = 1 / (1 + std::exp(-(input_data[i])));
        }
      },
      0, input_size, 1);
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
