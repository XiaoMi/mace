// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_OPS_ACTIVATION_H_
#define MACE_OPS_ACTIVATION_H_

#include <string>

#include "mace/core/operator.h"
#include "mace/kernels/activation.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class ActivationOp : public Operator<D, T> {
 public:
  ActivationOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(kernels::StringToActivationType(
                     OperatorBase::GetOptionalArg<std::string>("activation",
                                                               "NOOP")),
                 static_cast<T>(
                     OperatorBase::GetOptionalArg<float>("max_limit", 0.0f))) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(0);
    const Tensor *alpha_tensor =
        this->InputSize() >= 2 ? this->Input(1) : nullptr;
    Tensor *output_tensor = this->Output(0);
    MACE_RETURN_IF_ERROR(output_tensor->ResizeLike(input_tensor));

    return functor_(input_tensor, alpha_tensor, output_tensor, future);
  }

 private:
  kernels::ActivationFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ACTIVATION_H_
