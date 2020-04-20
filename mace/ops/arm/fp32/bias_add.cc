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

#include "mace/ops/arm/base/bias_add.h"

namespace mace {
namespace ops {
namespace arm {

template<>
void BiasAdd<float>::Add1DimBias(
    utils::ThreadPool *thread_pool, const float *input_data,
    const float *bias_data, float *output_data, const index_t batch,
    const index_t channels, const index_t image_size) {
  const index_t block_count = image_size / 4;
  const index_t remain = image_size % 4;
  thread_pool->Compute2D([=](index_t start0, index_t end0, index_t step0,
                             index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      const index_t b_offset = b * channels;
      for (index_t c = start1; c < end1; c += step1) {
        const index_t offset = (b_offset + c) * image_size;
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

template<>
void BiasAdd<float>::Add2DimsBias(
    utils::ThreadPool *thread_pool, const float *input_data,
    const float *bias_data, float *output_data, const index_t batch,
    const index_t channels, const index_t image_size) {
  const index_t block_count = image_size / 4;
  const index_t remain = image_size % 4;
  thread_pool->Compute2D([=](index_t start0, index_t end0, index_t step0,
                             index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      const index_t b_offset = b * channels;
      for (index_t c = start1; c < end1; c += step1) {
        const index_t offset = (b_offset + c) * image_size;
        auto input_ptr = input_data + offset;
        auto output_ptr = output_data + offset;
        const float bias = bias_data[b * channels + c];
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

}  // namespace arm
}  // namespace ops
}  // namespace mace
