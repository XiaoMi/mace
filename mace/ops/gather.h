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

#ifndef MACE_OPS_GATHER_H_
#define MACE_OPS_GATHER_H_

#include "mace/core/operator.h"
#include "mace/kernels/gather.h"

namespace mace {
namespace ops {

template<DeviceType D, class T>
class GatherOp : public Operator<D, T> {
 public:
  GatherOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(OperatorBase::GetOptionalArg<int>("axis", 0),
                 OperatorBase::GetOptionalArg<float>("y", 1.0)) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *params = this->Input(PARAMS);
    const Tensor *indices = this->Input(INDICES);
    Tensor *output = this->Output(OUTPUT);

    return functor_(params, indices, output, future);
  }

 private:
  kernels::GatherFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(PARAMS, INDICES);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_GATHER_H_
