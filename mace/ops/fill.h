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

#ifndef MACE_OPS_FILL_H_
#define MACE_OPS_FILL_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/fill.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class FillOp : public Operator<D, T> {
 public:
  FillOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_() {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *shape = this->Input(SHAPE);
    const Tensor *value = this->Input(VALUE);
    Tensor *output = this->Output(OUTPUT);
    return functor_(shape, value, output, future);
  }

 private:
  kernels::FillFunctor<D, T> functor_;

  MACE_OP_INPUT_TAGS(SHAPE, VALUE);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_FILL_H_
