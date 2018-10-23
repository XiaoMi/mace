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

#ifndef MACE_OPS_SQRDIFF_MEAN_H_
#define MACE_OPS_SQRDIFF_MEAN_H_

#include <string>
#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/sqrdiff_mean.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class SqrDiffMeanOp : public Operator<D, T> {
 public:
  SqrDiffMeanOp(const OperatorDef &operator_def, OpKernelContext *context)
      : Operator<D, T>(operator_def, context),
        functor_(context) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input0 = this->Input(INPUT0);
    const Tensor *input1 = this->Input(INPUT1);
    Tensor *output = this->Output(OUTPUT);

    return functor_(input0, input1, output, future);
  }

 private:
  kernels::SqrDiffMeanFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT0, INPUT1);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SQRDIFF_MEAN_H_
