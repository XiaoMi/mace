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

#ifndef MACE_OPS_ADDN_H_
#define MACE_OPS_ADDN_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/addn.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class AddNOp : public Operator<D, T> {
 public:
  AddNOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws) {}

  MaceStatus Run(StatsFuture *future) override {
    Tensor *output_tensor = this->Output(0);
    int n = this->inputs_.size();
    std::vector<const Tensor *> inputs(n, nullptr);
    inputs[0] = this->Input(0);
    for (int i = 1; i < n; ++i) {
      inputs[i] = this->Input(i);
      MACE_CHECK(inputs[0]->dim_size() == inputs[i]->dim_size());
      MACE_CHECK(inputs[0]->size() == inputs[i]->size())
          << "Input 0: " << MakeString(inputs[0]->shape())
          << ", size: " << inputs[0]->size() << ". Input " << i << ": "
          << MakeString(inputs[i]->shape()) << ", size: " << inputs[i]->size();
    }
    return functor_(inputs, output_tensor, future);
  }

 private:
  kernels::AddNFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ADDN_H_
