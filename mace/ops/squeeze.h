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

#ifndef MACE_OPS_SQUEEZE_H_
#define MACE_OPS_SQUEEZE_H_

#include <vector>
#include <unordered_set>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class SqueezeOp : public Operator<D, T> {
 public:
  SqueezeOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        axis_(OperatorBase::GetRepeatedArgs<int>("axis", {})) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);

    std::vector<index_t> output_shape;
    std::unordered_set<int> axis_set(axis_.begin(), axis_.end());
    for (int i = 0; i < input->dim_size(); ++i) {
      if (input->dim(i) > 1
          || (!axis_set.empty() && axis_set.find(i) == axis_set.end())) {
        output_shape.push_back(input->dim(i));
      }
    }
    output->ReuseTensorBuffer(*input);
    output->Reshape(output_shape);

    SetFutureDefaultWaitFn(future);
    return MACE_SUCCESS;
  }

 private:
  std::vector<int> axis_;

 private:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SQUEEZE_H_
