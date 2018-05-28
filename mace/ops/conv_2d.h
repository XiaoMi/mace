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

#ifndef MACE_OPS_CONV_2D_H_
#define MACE_OPS_CONV_2D_H_

#include <memory>
#include <string>

#include "mace/core/operator.h"
#include "mace/kernels/conv_2d.h"
#include "mace/ops/conv_pool_2d_base.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class Conv2dOp : public ConvPool2dOpBase<D, T> {
 public:
  Conv2dOp(const OperatorDef &op_def, Workspace *ws)
      : ConvPool2dOpBase<D, T>(op_def, ws),
        functor_(this->strides_.data(),
                 this->padding_type_,
                 this->paddings_,
                 this->dilations_.data(),
                 kernels::StringToActivationType(
                     OperatorBase::GetOptionalArg<std::string>("activation",
                                                               "NOOP")),
                 OperatorBase::GetOptionalArg<float>("max_limit", 0.0f),
                 static_cast<bool>(OperatorBase::GetOptionalArg<int>(
                     "is_filter_transformed", false)),
                 ws->GetScratchBuffer(D)) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);
    return functor_(input, filter, bias, output, future);
  }

 private:
  kernels::Conv2dFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CONV_2D_H_
