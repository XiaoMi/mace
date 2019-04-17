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

#ifndef MACE_OPS_ARM_FP32_ACTIVATION_H_
#define MACE_OPS_ARM_FP32_ACTIVATION_H_

#include "mace/core/op_context.h"
#include "mace/ops/common/activation_type.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

class Activation {
 public:
  explicit Activation(ActivationType type,
                      const float limit,
                      const float leakyrelu_coefficient);
  ~Activation() = default;

  MaceStatus Compute(
      const OpContext *context,
      const Tensor *input,
      Tensor *output);

 private:
  void DoActivation(const OpContext *context,
                    const Tensor *input,
                    Tensor *output);

  ActivationType type_;
  const float limit_;
  const float leakyrelu_coefficient_;
};

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FP32_ACTIVATION_H_
