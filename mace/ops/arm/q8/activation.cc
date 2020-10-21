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

#include <arm_neon.h>
#include <algorithm>

#include "mace/core/quantize.h"
#include "mace/ops/arm/base/activation.h"

namespace mace {
namespace ops {
namespace arm {

template<>
void Activation<uint8_t>::ActivateRelu(utils::ThreadPool *thread_pool,
                                       const Tensor *input,
                                       Tensor *output) {
  output->SetScale(input->scale());
  output->SetZeroPoint(input->zero_point());
  const auto input_data = input->data<uint8_t>();
  auto output_data = output->mutable_data<uint8_t>();
  const index_t input_size = input->size();
  const uint8x16_t vzero = vdupq_n_u8(input->zero_point());
  const index_t block_count = input_size / 16;

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        auto input_ptr = input_data + start * 16;
        auto output_ptr = output_data + start * 16;

        for (index_t i = start; i < end; i += step) {
          uint8x16_t v = vld1q_u8(input_ptr);
          v = vmaxq_u8(v, vzero);
          vst1q_u8(output_ptr, v);

          input_ptr += 16;
          output_ptr += 16;
        }
      }, 0, block_count, 1);

  // remain
  for (index_t i = block_count * 16; i < input_size; ++i) {
    output_data[i] = std::max<uint8_t>(input->zero_point(), input_data[i]);
  }
}

template<>
void Activation<uint8_t>::ActivateRelux(utils::ThreadPool *thread_pool,
                                        const Tensor *input,
                                        Tensor *output) {
  output->SetScale(input->scale());
  output->SetZeroPoint(input->zero_point());
  const auto input_data = input->data<uint8_t>();
  auto output_data = output->mutable_data<uint8_t>();
  const index_t input_size = input->size();
  const uint8x16_t vzero = vdupq_n_u8(input->zero_point());
  const uint8_t limit =
      Quantize<uint8_t>(limit_, input->scale(), input->zero_point());
  const uint8x16_t vlimit = vdupq_n_u8(limit);
  const index_t block_count = input_size / 16;

  thread_pool->Compute1D(
      [=](index_t start, index_t end, index_t step) {
        auto input_ptr = input_data + start * 16;
        auto output_ptr = output_data + start * 16;

        for (index_t i = start; i < end; i += step) {
          uint8x16_t v = vld1q_u8(input_ptr);
          v = vmaxq_u8(v, vzero);
          v = vminq_u8(v, vlimit);
          vst1q_u8(output_ptr, v);

          input_ptr += 16;
          output_ptr += 16;
        }
      }, 0, block_count, 1);

  // remain
  for (index_t i = block_count * 16; i < input_size; ++i) {
    output_data[i] =
        std::max<uint8_t>(input->zero_point(), std::min(limit, input_data[i]));
  }
}

template<>
void Activation<uint8_t>::ActivateLeakyRelu(utils::ThreadPool *thread_pool,
                                            const Tensor *input,
                                            Tensor *output) {
  MACE_UNUSED(thread_pool);
  MACE_UNUSED(input);
  MACE_UNUSED(output);
  MACE_NOT_IMPLEMENTED;
}

template<>
void Activation<uint8_t>::ActivateTanh(utils::ThreadPool *thread_pool,
                                       const Tensor *input,
                                       Tensor *output) {
  MACE_UNUSED(thread_pool);
  MACE_UNUSED(input);
  MACE_UNUSED(output);
  MACE_NOT_IMPLEMENTED;
}

template<>
void Activation<uint8_t>::ActivateSigmoid(utils::ThreadPool *thread_pool,
                                          const Tensor *input,
                                          Tensor *output) {
  MACE_UNUSED(thread_pool);
  MACE_UNUSED(input);
  MACE_UNUSED(output);
  MACE_NOT_IMPLEMENTED;
}

template<>
void Activation<uint8_t>::ActivateElu(utils::ThreadPool *thread_pool,
                                      const Tensor *input,
                                      Tensor *output) {
  MACE_UNUSED(thread_pool);
  MACE_UNUSED(input);
  MACE_UNUSED(output);
  MACE_NOT_IMPLEMENTED;
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
