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

#ifndef MACE_OPS_DECONV_2D_H_
#define MACE_OPS_DECONV_2D_H_

#include <memory>
#include <string>

#include "mace/core/operator.h"
#include "mace/kernels/deconv_2d.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class Deconv2dOp : public Operator<D, T> {
 public:
  Deconv2dOp(const OperatorDef &op_def, OpKernelContext *context)
      : Operator<D, T>(op_def, context),
        functor_(context,
                 OperatorBase::GetRepeatedArgs<int>("strides"),
                 static_cast<Padding>(OperatorBase::GetOptionalArg<int>(
                     "padding", static_cast<int>(SAME))),
                 OperatorBase::GetRepeatedArgs<int>("padding_values"),
                 static_cast<kernels::FrameworkType>(
                     OperatorBase::GetOptionalArg<int>("framework_type", 0)),
                 kernels::StringToActivationType(
                     OperatorBase::GetOptionalArg<std::string>("activation",
                                                               "NOOP")),
                 OperatorBase::GetOptionalArg<float>("max_limit", 0.0f)) {}

  MaceStatus Run(StatsFuture *future) override {
    MACE_CHECK(this->InputSize() >= 2, "deconv needs >= 2 inputs.");
    const Tensor *input = this->Input(0);
    const Tensor *filter = this->Input(1);
    kernels::FrameworkType model_type =
        static_cast<kernels::FrameworkType>(
            OperatorBase::GetOptionalArg<int>("framework_type", 0));
    if (model_type == kernels::CAFFE) {
      const Tensor *bias = this->InputSize() >= 3 ? this->Input(2) : nullptr;
      Tensor *output = this->Output(OUTPUT);

      return functor_(input, filter, bias, nullptr, output, future);
    } else {
      const Tensor *output_shape =
          this->InputSize() >= 3 ? this->Input(2) : nullptr;
      const Tensor *bias = this->InputSize() >= 4 ? this->Input(3) : nullptr;
      Tensor *output = this->Output(OUTPUT);

      return functor_(input, filter, bias, output_shape, output, future);
    }
  }

 private:
  kernels::Deconv2dFunctor<D, T> functor_;

 protected:
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DECONV_2D_H_
