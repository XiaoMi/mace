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

#include "mace/ops/common/gemmlowp_util.h"
#include "mace/ops/delegator/eltwise.h"
#include "mace/utils/logging.h"

namespace mace {
namespace ops {
namespace ref {
namespace q8 {

class Eltwise : public delegator::Eltwise {
 public:
  explicit Eltwise(const delegator::EltwiseParam &param)
      : delegator::Eltwise(param) {}
  ~Eltwise() = default;

  MaceStatus Compute(const OpContext *context, const Tensor *input0,
                     const Tensor *input1, Tensor *output) override;
};

MaceStatus Eltwise::Compute(const OpContext *context,
                            const Tensor *input0,
                            const Tensor *input1,
                            Tensor *output) {
  constexpr int left_shift = 20;
  const double doubled_scale = 2 * std::max(input0->scale(), input1->scale());
  const double adjusted_input0_scale = input0->scale() / doubled_scale;
  const double adjusted_input1_scale = input1->scale() / doubled_scale;
  const double adjusted_output_scale =
      doubled_scale / ((1 << left_shift) * output->scale());

  int32_t input0_multiplier;
  int32_t input1_multiplier;
  int32_t output_multiplier;
  int32_t input0_shift;
  int32_t input1_shift;
  int32_t output_shift;
  QuantizeMultiplier(adjusted_input0_scale,
                     &input0_multiplier,
                     &input0_shift);
  QuantizeMultiplier(adjusted_input1_scale,
                     &input1_multiplier,
                     &input1_shift);
  QuantizeMultiplier(adjusted_output_scale,
                     &output_multiplier,
                     &output_shift);

  Tensor::MappingGuard input0_guard(input0);
  Tensor::MappingGuard input1_guard(input1);
  Tensor::MappingGuard output_guard(output);

  auto input0_ptr = input0->data<uint8_t>();
  auto input1_ptr = input1->data<uint8_t>();
  auto output_ptr = output->mutable_data<uint8_t>();

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();
  thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
    for (index_t i = start; i < end; i += step) {
      const int32_t offset_input0 = input0_ptr[i] - input0->zero_point();
      const int32_t offset_input1 = input1_ptr[i] - input1->zero_point();
      const int32_t shifted_input0 = offset_input0 * (1 << left_shift);
      const int32_t shifted_input1 = offset_input1 * (1 << left_shift);
      const int32_t multiplied_input0 =
          gemmlowp::RoundingDivideByPOT(
              gemmlowp::SaturatingRoundingDoublingHighMul(shifted_input0,
                                                          input0_multiplier),
              -input0_shift);
      const int32_t multiplied_input1 =
          gemmlowp::RoundingDivideByPOT(
              gemmlowp::SaturatingRoundingDoublingHighMul(shifted_input1,
                                                          input1_multiplier),
              -input1_shift);

      int32_t res;
      if (type_ == SUM) {
        res = multiplied_input0 + multiplied_input1;
      } else {
        res = multiplied_input0 - multiplied_input1;
      }

      const int32_t output_val =
          gemmlowp::RoundingDivideByPOT(
              gemmlowp::SaturatingRoundingDoublingHighMul(res,
                                                          output_multiplier),
              -output_shift) + output->zero_point();
      output_ptr[i] = Saturate<uint8_t>(output_val);
    }
  }, 0, output->size(), 1);

  return MaceStatus::MACE_SUCCESS;
}

void RegisterEltwiseDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Eltwise, delegator::EltwiseParam,
      MACE_DELEGATOR_KEY(Eltwise, DeviceType::CPU, uint8_t, ImplType::REF));
}

}  // namespace q8
}  // namespace ref
}  // namespace ops
}  // namespace mace
