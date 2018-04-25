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

template<DeviceType D, class T>
class FullyConnectedOp : public Operator<D, T> {
 public:
  FullyConnectedOp(const OperatorDef &operator_def, Workspace *ws)
    : Operator<D, T>(operator_def, ws),
      functor_(static_cast<kernels::BufferType>(
                 OperatorBase::GetSingleArgument<int>(
                   "weight_type", static_cast<int>(
                     kernels::WEIGHT_WIDTH))),
               kernels::StringToActivationType(
                 OperatorBase::GetSingleArgument<std::string>("activation",
                                                              "NOOP")),
               OperatorBase::GetSingleArgument<float>("max_limit", 0.0f)) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *weight = this->Input(WEIGHT);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    const index_t input_size = input->dim(1) * input->dim(2) * input->dim(3);
    MACE_CHECK(input_size == weight->dim(1) && weight->dim(0) == bias->dim(0),
               "The size of Input: ",
               input_size,
               " Weight: ",
               weight->dim(1),
               ",",
               weight->dim(
                 0),
               " and Bias ",
               bias->dim(0),
               " don't match.");

    functor_(input, weight, bias, output, future);
    return true;
  }

 private:
  kernels::FullyConnectedFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, WEIGHT, BIAS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_FULLY_CONNECTED_H_
