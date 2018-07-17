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

#ifndef MACE_OPS_CROP_H_
#define MACE_OPS_CROP_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/crop.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class CropOp : public Operator<D, T> {
 public:
  CropOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(OperatorBase::GetOptionalArg<int>("axis", 2),
                 OperatorBase::GetRepeatedArgs<int>("offset")) {}

  MaceStatus Run(StatsFuture *future) override {
    MACE_CHECK(this->InputSize() >= 2)
        << "There must be two inputs to crop";
    const std::vector<const Tensor *> input_list = this->Inputs();
    Tensor *output = this->Output(0);
    return functor_(input_list, output, future);
  }

 private:
  kernels::CropFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CROP_H_
