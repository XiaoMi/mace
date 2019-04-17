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


#include "mace/ops/ref/gemv.h"

#if defined(MACE_ENABLE_QUANTIZE)
#include "mace/core/quantize.h"
#endif  // MACE_ENABLE_QUANTIZE

namespace mace {
namespace ops {
namespace ref {

MaceStatus Gemv<float>::Compute(const OpContext *context,
                                const Tensor *lhs,
                                const Tensor *rhs,
                                const Tensor *bias,
                                const index_t batch,
                                const index_t lhs_height,
                                const index_t lhs_width,
                                const bool lhs_batched,
                                const bool rhs_batched,
                                Tensor *output) {
  MACE_UNUSED(context);

  Tensor::MappingGuard lhs_guard(lhs);
  Tensor::MappingGuard rhs_guard(rhs);
  Tensor::MappingGuard bias_guard(bias);
  Tensor::MappingGuard output_guard(output);
  const float *lhs_data = lhs->data<float>();
  const float *rhs_data = rhs->data<float>();
  const float *bias_data = nullptr;
  if (bias) {
    bias_data = bias->data<float>();
  }

  float *output_data = output->mutable_data<float>();

  for (index_t b = 0; b < batch; ++b) {
    for (index_t h = 0; h < lhs_height; ++h) {
      float sum = bias ? bias_data[h] : 0;
      for (index_t w = 0; w < lhs_width; ++w) {
        sum += lhs_data[
            static_cast<index_t>(lhs_batched) * b * lhs_height * lhs_width
                + h * lhs_width + w]
            * rhs_data[static_cast<index_t>(rhs_batched) * b * lhs_width + w];
      }  // w

      output_data[b * lhs_height + h] = sum;
    }  // h
  }   // b

  return MaceStatus::MACE_SUCCESS;
}

#if defined(MACE_ENABLE_QUANTIZE)
MaceStatus Gemv<uint8_t>::Compute(const OpContext *context,
                                  const Tensor *lhs,
                                  const Tensor *rhs,
                                  const Tensor *bias,
                                  const index_t batch,
                                  const index_t lhs_height,
                                  const index_t lhs_width,
                                  const bool lhs_batched,
                                  const bool rhs_batched,
                                  Tensor *output) {
  MACE_UNUSED(context);

  Tensor::MappingGuard lhs_guard(lhs);
  Tensor::MappingGuard rhs_guard(rhs);
  Tensor::MappingGuard bias_guard(bias);
  Tensor::MappingGuard output_guard(output);
  const uint8_t *lhs_data = lhs->data<uint8_t>();
  const uint8_t *rhs_data = rhs->data<uint8_t>();
  const int32_t *bias_data = nullptr;
  if (bias) {
    bias_data = bias->data<int32_t>();
  }

  uint8_t *output_data = output->mutable_data<uint8_t>();

  MACE_CHECK(output->scale() > 0, "output scale must not be zero");
  const float
      output_multiplier_float = lhs->scale() * rhs->scale() / output->scale();
  int32_t lhs_zero = lhs->zero_point();
  int32_t rhs_zero = rhs->zero_point();

  for (index_t b = 0; b < batch; ++b) {
    for (index_t h = 0; h < lhs_height; ++h) {
      int32_t sum = bias ? bias_data[h] : 0;
      for (index_t w = 0; w < lhs_width; ++w) {
        sum += (lhs_data[
            static_cast<index_t>(lhs_batched) * b * lhs_height * lhs_width
                + h * lhs_width + w] - lhs_zero)
            * (rhs_data[static_cast<index_t>(rhs_batched) * b * lhs_width + w]
                - rhs_zero);
      }  // w

      output_data[b * lhs_height + h] =
          Saturate<uint8_t>(std::roundf(sum * output_multiplier_float));
    }  // h
  }   // b
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Gemv<int32_t>::Compute(const OpContext *context,
                                  const Tensor *lhs,
                                  const Tensor *rhs,
                                  const Tensor *bias,
                                  const index_t batch,
                                  const index_t lhs_height,
                                  const index_t lhs_width,
                                  const bool lhs_batched,
                                  const bool rhs_batched,
                                  Tensor *output) {
  MACE_UNUSED(context);

  Tensor::MappingGuard lhs_guard(lhs);
  Tensor::MappingGuard rhs_guard(rhs);
  Tensor::MappingGuard bias_guard(bias);
  Tensor::MappingGuard output_guard(output);
  const uint8_t *lhs_data = lhs->data<uint8_t>();
  const uint8_t *rhs_data = rhs->data<uint8_t>();
  const int32_t *bias_data = nullptr;
  if (bias) {
    bias_data = bias->data<int32_t>();
  }

  int32_t *output_data = output->mutable_data<int32_t>();

  int32_t lhs_zero = lhs->zero_point();
  int32_t rhs_zero = rhs->zero_point();

  for (index_t b = 0; b < batch; ++b) {
    for (index_t h = 0; h < lhs_height; ++h) {
      int32_t sum = bias ? bias_data[h] : 0;
      for (index_t w = 0; w < lhs_width; ++w) {
        sum += (lhs_data[
            static_cast<index_t>(lhs_batched) * b * lhs_height * lhs_width
                + h * lhs_width + w] - lhs_zero)
            * (rhs_data[static_cast<index_t>(rhs_batched) * b * lhs_width + w]
                - rhs_zero);
      }  // w

      output_data[b * lhs_height + h] = sum;
    }  // h
  }   // b
  return MaceStatus::MACE_SUCCESS;
}
#endif  // MACE_ENABLE_QUANTIZE

}  // namespace ref
}  // namespace ops
}  // namespace mace
