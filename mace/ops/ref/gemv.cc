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


#include "mace/ops/delegator/gemv.h"

#if defined(MACE_ENABLE_QUANTIZE)
#include "mace/core/quantize.h"
#endif  // MACE_ENABLE_QUANTIZE

namespace mace {
namespace ops {
namespace ref {

template<typename T>
class Gemv : public delegator::Gemv {
 public:
  explicit Gemv(const DelegatorParam &param) : delegator::Gemv(param) {}
  ~Gemv() {}
  // Always row-major after transpose
  MaceStatus Compute(
      const OpContext *context,
      const Tensor *lhs,
      const Tensor *rhs,
      const Tensor *bias,
      const index_t batch,
      const index_t lhs_height,
      const index_t lhs_width,
      const bool lhs_batched,
      const bool rhs_batched,
      Tensor *output) override;
};

template<typename T>
MaceStatus Gemv<T>::Compute(const OpContext *context,
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
  const T *lhs_data = lhs->data<T>();
  const T *rhs_data = rhs->data<T>();
  const T *bias_data = nullptr;
  if (bias) {
    bias_data = bias->data<T>();
  }

  T *output_data = output->mutable_data<T>();

  for (index_t b = 0; b < batch; ++b) {
    for (index_t h = 0; h < lhs_height; ++h) {
      float sum = bias ? static_cast<float>(bias_data[h]) : 0.f;
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

void RegisterGemvDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Gemv<float>, DelegatorParam,
      MACE_DELEGATOR_KEY(Gemv, DeviceType::CPU, float, ImplType::REF));
  MACE_REGISTER_BF16_DELEGATOR(
      registry, Gemv<BFloat16>, DelegatorParam,
      MACE_DELEGATOR_KEY(Gemv, DeviceType::CPU, BFloat16, ImplType::REF));
}

}  // namespace ref
}  // namespace ops
}  // namespace mace
