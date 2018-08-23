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
        functor_(
            static_cast<kernels::EltwiseType>(OperatorBase::GetOptionalArg<int>(
                "type", static_cast<int>(kernels::EltwiseType::NONE))),
            OperatorBase::GetRepeatedArgs<float>("coeff"),
            OperatorBase::GetOptionalArg<float>("scalar_input", 1.0),
            OperatorBase::GetOptionalArg<int32_t>("scalar_input_index", 1),
            static_cast<DataFormat>(OperatorBase::GetOptionalArg<int>(
                "data_format", 0))) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input0 = this->Input(0);
    const Tensor *input1 = this->InputSize() == 2 ? this->Input(1) : nullptr;
    Tensor *output = this->Output(OUTPUT);
    return functor_(input0, input1, output, future);
  }

 private:
  kernels::EltwiseFunctor<D, T> functor_;

 private:
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ELTWISE_H_
