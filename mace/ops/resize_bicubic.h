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

#ifndef MACE_OPS_RESIZE_BICUBIC_H_
#define MACE_OPS_RESIZE_BICUBIC_H_

#include "mace/core/operator.h"
#include "mace/kernels/resize_bicubic.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class ResizeBicubicOp : public Operator<D, T> {
 public:
  ResizeBicubicOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(OperatorBase::GetRepeatedArgs<index_t>("size", {-1, -1}),
                 OperatorBase::GetOptionalArg<bool>("align_corners", false)) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional.",
               input->dim_size());

    return functor_(input, output, future);
  }

 private:
  kernels::ResizeBicubicFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_RESIZE_BICUBIC_H_

