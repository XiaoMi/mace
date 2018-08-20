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

#ifndef MACE_OPS_SPLIT_H_
#define MACE_OPS_SPLIT_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/split.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class SplitOp : public Operator<D, T> {
 public:
  SplitOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(OperatorBase::GetOptionalArg<int>("axis", 3)) {}

  MaceStatus Run(StatsFuture *future) override {
    MACE_CHECK(this->OutputSize() >= 2)
        << "There must be at least two outputs for slicing";
    const Tensor *input = this->Input(INPUT);
    const std::vector<Tensor *> output_list = this->Outputs();
    const int32_t split_axis = OperatorBase::GetOptionalArg<int>("axis", 3);
    MACE_CHECK((input->dim(split_axis) % this->OutputSize()) == 0)
        << "Outputs do not split input equally.";

    return functor_(input, output_list, future);
  }

 private:
  kernels::SplitFunctor<D, T> functor_;

 private:
  MACE_OP_INPUT_TAGS(INPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SPLIT_H_
