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

#ifndef MACE_OPS_PAD_H_
#define MACE_OPS_PAD_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/pad.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class PadOp : public Operator<D, T> {
 public:
  PadOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(OperatorBase::GetRepeatedArgs<int>("paddings"),
                 OperatorBase::GetOptionalArg<float>("constant_value", 0.0)) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(0);
    Tensor *output_tensor = this->Output(0);
    return functor_(input_tensor, output_tensor, future);
  }

 private:
  kernels::PadFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_PAD_H_
