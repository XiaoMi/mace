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

#include "mace/core/quantize.h"
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
      float real_input0 =
          input0->scale() * (input0_ptr[i] - input0->zero_point());
      float real_input1 =
          input1->scale() * (input1_ptr[i] - input1->zero_point());
      int32_t res;
      if (type_ == SUM) {
        res = real_input0 + real_input1;
      } else {
        res = real_input0 - real_input1;
      }

      output_ptr[i] =
          Quantize<uint8_t>(res, output->scale(), output->zero_point());
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
