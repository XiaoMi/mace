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

#include "mace/ops/arm/fp32/bias_add.h"

#include <arm_neon.h>

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

MaceStatus BiasAdd::Compute(const OpContext *context,
                            const Tensor *input,
                            const Tensor *bias,
                            Tensor *output) {
  Tensor::MappingGuard input_guard(input);
  Tensor::MappingGuard bias_guard(bias);
  if (input != output) {
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    if (bias == nullptr) {
      output->Copy(*input);
    } else {
      Tensor::MappingGuard output_guard(output);
      AddBias(context, input, bias, output);
    }
  } else {
    if (bias != nullptr) {
      AddBias(context, input, bias, output);
    }
  }

  return MaceStatus::MACE_SUCCESS;
}

void BiasAdd::AddBias(const OpContext *context,
                      const Tensor *input,
                      const Tensor *bias,
                      mace::Tensor *output) {
  auto input_data = input->data<float>();
  auto bias_data = bias->data<float>();
  auto output_data = output->mutable_data<float>();

  const index_t batch = input->dim(0);
  const index_t channels = input->dim(1);
  const index_t height = output->dim(2);
  const index_t width = output->dim(3);
  const index_t image_size = height * width;
  const index_t block_count = image_size / 4;
  const index_t remain = image_size % 4;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();
  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t c = start1; c < end1; c += step1) {
        const index_t offset = (b * channels + c) * image_size;
        auto input_ptr = input_data + offset;
        auto output_ptr = output_data + offset;
        const float bias = bias_data[c];
        float32x4_t vbias = vdupq_n_f32(bias);

        for (index_t i = 0; i < block_count; ++i) {
          float32x4_t v = vld1q_f32(input_ptr);
          v = vaddq_f32(v, vbias);
          vst1q_f32(output_ptr, v);

          input_ptr += 4;
          output_ptr += 4;
        }
        for (index_t i = 0; i < remain; ++i) {
          (*output_ptr++) = (*input_ptr++) + bias;
        }
      }
    }
  }, 0, batch, 1, 0, channels, 1);
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace

