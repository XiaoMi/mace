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

#ifndef MACE_OPS_CONCAT_H_
#define MACE_OPS_CONCAT_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/concat.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class ConcatOp : public Operator<D, T> {
 public:
  ConcatOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(OperatorBase::GetOptionalArg<int>("axis", 3)) {}

  MaceStatus Run(StatsFuture *future) override {
    MACE_CHECK(this->InputSize() >= 2)
        << "There must be at least two inputs to concat";
    const std::vector<const Tensor *> input_list = this->Inputs();
    const int32_t concat_axis = OperatorBase::GetOptionalArg<int>("axis", 3);
    const int32_t input_dims = input_list[0]->dim_size();
    const int32_t axis =
        concat_axis < 0 ? concat_axis + input_dims : concat_axis;
    MACE_CHECK((0 <= axis && axis < input_dims),
               "Expected concatenating axis in the range [", -input_dims, ", ",
               input_dims, "], but got", concat_axis);

    Tensor *output = this->Output(OUTPUT);

    return functor_(input_list, output, future);
  }

 private:
  kernels::ConcatFunctor<D, T> functor_;

 private:
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CONCAT_H_
