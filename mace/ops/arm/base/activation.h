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

#ifndef MACE_OPS_ARM_BASE_ACTIVATION_H_
#define MACE_OPS_ARM_BASE_ACTIVATION_H_

#include "mace/ops/delegator/activation.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class Activation : public delegator::Activation {
 public:
  explicit Activation(const delegator::ActivationParam &param)
      : delegator::Activation(param) {}
  ~Activation() = default;

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input, Tensor *output) override;

 private:
  void DoActivation(const OpContext *context,
                    const Tensor *input, Tensor *output);

  void ActivateRelu(utils::ThreadPool *thread_pool, const Tensor *input,
                    Tensor *output);
  void ActivateRelux(utils::ThreadPool *thread_pool, const Tensor *input,
                     Tensor *output);
  void ActivateLeakyRelu(utils::ThreadPool *thread_pool, const Tensor *input,
                         Tensor *output);
  void ActivateTanh(utils::ThreadPool *thread_pool, const Tensor *input,
                    Tensor *output);
  void ActivateSigmoid(utils::ThreadPool *thread_pool, const Tensor *input,
                       Tensor *output);
  void ActivateElu(utils::ThreadPool *thread_pool, const Tensor *input,
                   Tensor *output);
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_ACTIVATION_H_
