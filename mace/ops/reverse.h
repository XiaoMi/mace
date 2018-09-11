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

#ifndef MACE_OPS_REVERSE_H_
#define MACE_OPS_REVERSE_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/reverse.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class ReverseOp : public Operator<D, T> {
 public:
  ReverseOp(const OperatorDef &operator_def, OpKernelContext *context)
      : Operator<D, T>(operator_def, context), functor_(context) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *axis = this->Input(AXIS);
    Tensor *output = this->Output(OUTPUT);
    return functor_(input, axis, output, future);
  }

 private:
  kernels::ReverseFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, AXIS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_REVERSE_H_
