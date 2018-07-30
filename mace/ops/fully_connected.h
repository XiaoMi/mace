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

#ifndef MACE_OPS_FULLY_CONNECTED_H_
#define MACE_OPS_FULLY_CONNECTED_H_

#include <string>

#include "mace/core/operator.h"
#include "mace/kernels/fully_connected.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class FullyConnectedOp : public Operator<D, T> {
 public:
  FullyConnectedOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(kernels::StringToActivationType(
                     OperatorBase::GetOptionalArg<std::string>("activation",
                                                               "NOOP")),
                 OperatorBase::GetOptionalArg<float>("max_limit", 0.0f)) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *weight = this->Input(WEIGHT);  // OIHW
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    if (D == DeviceType::CPU) {
      MACE_CHECK(
          input->dim(1) == weight->dim(1) && input->dim(2) == weight->dim(2) &&
              input->dim(3) == weight->dim(3),
          "The shape of Input: ", MakeString(input->shape()),
          "The shape of Weight: ", MakeString(weight->shape()),
          " don't match.");
    } else {
      MACE_CHECK(
          input->dim(1) == weight->dim(2) && input->dim(2) == weight->dim(3) &&
              input->dim(3) == weight->dim(1),
          "The shape of Input: ", MakeString(input->shape()),
          "The shape of Weight: ", MakeString(weight->shape()),
          " don't match.");
    }
    if (bias) {
      MACE_CHECK(weight->dim(0) == bias->dim(0),
                 "The shape of Weight: ", MakeString(weight->shape()),
                 " and shape of Bias: ", bias->dim(0),
                 " don't match.");
    }

    return functor_(input, weight, bias, output, future);
  }

 private:
  kernels::FullyConnectedFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, WEIGHT, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_FULLY_CONNECTED_H_
