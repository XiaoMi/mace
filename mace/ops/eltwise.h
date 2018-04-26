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

#ifndef MACE_OPS_ELTWISE_H_
#define MACE_OPS_ELTWISE_H_

#include "mace/core/operator.h"
#include "mace/kernels/eltwise.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class EltwiseOp : public Operator<D, T> {
 public:
  EltwiseOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(static_cast<kernels::EltwiseType>(
                     OperatorBase::GetSingleArgument<int>(
                         "type", static_cast<int>(kernels::EltwiseType::SUM))),
                 OperatorBase::GetRepeatedArgument<float>("coeff")) {}

  bool Run(StatsFuture *future) override {
    if (this->InputSize() == 1) {
      const Tensor* input = this->Input(0);
      Tensor *output = this->Output(OUTPUT);
      start_axis_ = input->dim_size() - 1;
      is_scaler_ = true;
      output->ResizeLike(input);
      const float x = OperatorBase::GetSingleArgument<float>("x", 1.0);
      functor_(input, nullptr, start_axis_,
               is_scaler_, x, false, output, future);
    } else {
      const index_t size0 = this->Input(0)->size();
      const index_t size1 = this->Input(1)->size();
      const bool swap = (size0 < size1);
      const Tensor *input0 = swap ? this->Input(1) : this->Input(0);
      const Tensor *input1 = swap ? this->Input(0) : this->Input(1);

      Tensor *output = this->Output(OUTPUT);
      MACE_CHECK(input0->dim_size() == input1->dim_size())
        << "Inputs of Eltwise op must be same shape";
      start_axis_ = input0->dim_size() - 1;
      is_scaler_ = (input1->size() == 1);
      uint32_t compared_size = 1;
      if (!is_scaler_) {
        while (start_axis_ >= 0) {
          MACE_CHECK(input0->dim(start_axis_) == input1->dim(start_axis_),
                     "Invalid inputs dimension at axis: ") << start_axis_
                     << "input 0: " << input0->dim(start_axis_)
                     << "input 1: " << input1->dim(start_axis_);
          compared_size *= input1->dim(start_axis_);
          if (compared_size == input1->size()) {
            break;
          }
          start_axis_--;
        }
      }
      output->ResizeLike(input0);
      const float x = OperatorBase::GetSingleArgument<float>("x", 1.0);
      functor_(input0, input1, start_axis_,
               is_scaler_, x, swap, output, future);
    }
    return true;
  }

 private:
  kernels::EltwiseFunctor<D, T> functor_;
  index_t start_axis_;
  bool is_scaler_;

 private:
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ELTWISE_H_
