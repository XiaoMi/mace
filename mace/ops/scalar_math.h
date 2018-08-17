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

#ifndef MACE_OPS_SCALAR_MATH_H_
#define MACE_OPS_SCALAR_MATH_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/scalar_math.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class ScalarMathOp : public Operator<D, T> {
 public:
  ScalarMathOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(static_cast<kernels::EltwiseType>(
                   OperatorBase::GetOptionalArg<int>(
                       "type", static_cast<int>(kernels::EltwiseType::NONE))),
                 OperatorBase::GetRepeatedArgs<float>("coeff"),
                 OperatorBase::GetOptionalArg<float>("scalar_input", 1.0),
                 OperatorBase::GetOptionalArg<int32_t>(
                     "scalar_input_index", 1)) {}

  MaceStatus Run(StatsFuture *future) override {
    const std::vector<const Tensor *> input_list = this->Inputs();
    Tensor *output = this->Output(0);
    return functor_(input_list, output, future);
  }

 private:
  kernels::ScalarMathFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SCALAR_MATH_H_
