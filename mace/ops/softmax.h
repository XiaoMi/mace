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

#ifndef MACE_OPS_SOFTMAX_H_
#define MACE_OPS_SOFTMAX_H_

#include "mace/core/operator.h"
#include "mace/kernels/softmax.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class SoftmaxOp : public Operator<D, T> {
 public:
  SoftmaxOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *logits = this->Input(LOGITS);

    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(logits));

    return functor_(logits, output, future);
  }

 private:
  kernels::SoftmaxFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(LOGITS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SOFTMAX_H_
