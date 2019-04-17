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

#include "mace/ops/arm/fp32/activation.h"

#include <arm_neon.h>
#include <algorithm>

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

Activation::Activation(ActivationType type,
                       const float limit,
                       const float leakyrelu_coefficient)
    : type_(type),
      limit_(limit),
      leakyrelu_coefficient_(leakyrelu_coefficient) {}

MaceStatus Activation::Compute(const OpContext *context,
                               const Tensor *input,
                               Tensor *output) {
  Tensor::MappingGuard input_guard(input);
  if (input != output) {
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    Tensor::MappingGuard output_guard(output);
    DoActivation(context, input, output);
  } else {
    DoActivation(context, input, output);
  }

  return MaceStatus::MACE_SUCCESS;
}

void Activation::DoActivation(const OpContext *context,
                              const Tensor *input,
                              Tensor *output) {
  auto input_data = input->data<float>();
  auto output_data = output->mutable_data<float>();
  const index_t size = input->size();

  utils::ThreadPool &thread_pool =
      context->device()->cpu_runtime()->thread_pool();

  switch (type_) {
    case RELU: {
      const float32x4_t vzero = vdupq_n_f32(0.f);
      const index_t block_count = size / 4;

      thread_pool.Compute1D(
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
      for (index_t i = block_count * 4; i < size; ++i) {
        output_data[i] = std::max(0.f, input_data[i]);
      }

      break;
    }

    case RELUX: {
      const float32x4_t vzero = vdupq_n_f32(0.f);
      const float32x4_t vlimit = vdupq_n_f32(limit_);
      const index_t block_count = size / 4;

      thread_pool.Compute1D(
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
      for (index_t i = block_count * 4; i < size; ++i) {
        output_data[i] = std::max(0.f, std::min(limit_, input_data[i]));
      }

      break;
    }

    case LEAKYRELU: {
      const float32x4_t vzero = vdupq_n_f32(0.f);
      const float32x4_t valpha = vdupq_n_f32(leakyrelu_coefficient_);
      const index_t block_count = size / 4;

      thread_pool.Compute1D(
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
      for (index_t i = block_count * 4; i < size; ++i) {
        output_data[i] = std::max(input_data[i], 0.f) +
                         std::min(input_data[i], 0.f) * leakyrelu_coefficient_;
      }

      break;
    }

    case TANH: {
      thread_pool.Compute1D(
          [=](index_t start, index_t end, index_t step) {
            for (index_t i = start; i < end; i += step) {
              output_data[i] = std::tanh(input_data[i]);
            }
          },
          0, size, 1);

      break;
    }

    case SIGMOID: {
      thread_pool.Compute1D(
          [=](index_t start, index_t end, index_t step) {
            for (index_t i = start; i < end; i += step) {
              output_data[i] = 1 / (1 + std::exp(-(input_data[i])));
            }
          },
          0, size, 1);

      break;
    }

    case NOOP:
      break;

    default:
      MACE_NOT_IMPLEMENTED;
  }
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
