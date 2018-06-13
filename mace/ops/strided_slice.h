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

#ifndef MACE_OPS_STRIDED_SLICE_H_
#define MACE_OPS_STRIDED_SLICE_H_

#include "mace/core/operator.h"
#include "mace/kernels/strided_slice.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class StridedSliceOp : public Operator<D, T> {
 public:
  StridedSliceOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(OperatorBase::GetOptionalArg<int>("begin_mask", 0),
                 OperatorBase::GetOptionalArg<int>("end_mask", 0),
                 OperatorBase::GetOptionalArg<int>("ellipsis_mask", 0),
                 OperatorBase::GetOptionalArg<int>("new_axis_mask", 0),
                 OperatorBase::GetOptionalArg<int>("shrink_axis_mask", 0),
                 OperatorBase::GetOptionalArg<bool>("slice",
                                                    false)) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *begin_indices = this->Input(BEGIN);
    const Tensor *end_indices = this->Input(END);
    const Tensor *strides = nullptr;
    if (this->InputSize() > 3) {
      strides = this->Input(STRIDES);
    }
    Tensor *output = this->Output(OUTPUT);

    return functor_(input, begin_indices, end_indices, strides, output, future);
  }

 private:
  kernels::StridedSliceFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, BEGIN, END, STRIDES);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_STRIDED_SLICE_H_
