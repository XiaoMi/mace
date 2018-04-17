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

#ifndef MACE_OPS_CWISE_H_
#define MACE_OPS_CWISE_H_

#include <string>

#include "mace/core/operator.h"
#include "mace/kernels/cwise.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class CWiseOp : public Operator<D, T> {
 public:
  CWiseOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        x_(OperatorBase::GetSingleArgument<float>("x", 1.0)),
        functor_(static_cast<kernels::CWiseType>(
                     OperatorBase::GetSingleArgument<int>(
                         "type", static_cast<int>(
                             kernels::CWiseType::ADD))),
                 this->x_) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(INPUT);
    Tensor *output_tensor = this->Output(OUTPUT);
    output_tensor->ResizeLike(input_tensor);

    functor_(input_tensor, output_tensor, future);
    return true;
  }

 protected:
  const float x_;
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);

 private:
  kernels::CWiseFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CWISE_H_
