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

#ifndef MACE_OPS_ELTWISE_H_
#define MACE_OPS_ELTWISE_H_

#include "mace/core/operator.h"
#include "mace/kernels/eltwise.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class EltwiseOp : public Operator<D, T> {
 public:
  EltwiseOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(static_cast<kernels::EltwiseType>(
                     OperatorBase::GetSingleArgument<int>(
                         "type", static_cast<int>(kernels::EltwiseType::SUM))),
                 OperatorBase::GetRepeatedArgument<float>("coeff")) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input0 = this->Input(0);
    const Tensor *input1 = this->Input(1);
    Tensor *output = this->Output(OUTPUT);
    MACE_CHECK(input0->dim_size() == input1->dim_size())
        << "Inputs of Eltwise op must be same shape";
    for (int i = 0; i < input0->dim_size(); ++i) {
      MACE_CHECK(input0->dim(i) == input1->dim(i))
          << "Inputs of Eltwise op must be same shape";
    }

    output->ResizeLike(input0);

    functor_(input0, input1, output, future);
    return true;
  }

 private:
  kernels::EltwiseFunctor<D, T> functor_;

 private:
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ELTWISE_H_
