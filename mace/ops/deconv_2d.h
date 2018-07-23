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

#include "mace/core/operator.h"
#include "mace/kernels/deconv_2d.h"
#include "mace/ops/conv_pool_2d_base.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class Deconv2dOp : public ConvPool2dOpBase<D, T> {
 public:
  Deconv2dOp(const OperatorDef &op_def, Workspace *ws)
      : ConvPool2dOpBase<D, T>(op_def, ws),
        functor_(this->strides_.data(),
                 this->padding_type_,
                 this->paddings_,
                 OperatorBase::GetRepeatedArgs<index_t>("output_shape"),
                 kernels::ActivationType::NOOP,
                 0.0f,
                 OperatorBase::GetOptionalArg<bool>("from_caffe", false)) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *output_shape =
        this->InputSize() >= 3 ? this->Input(OUTPUT_SHAPE) : nullptr;
    const Tensor *bias = this->InputSize() >= 4 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    return functor_(input, filter, bias, output_shape, output, future);
  }

 private:
  kernels::Deconv2dFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, FILTER, OUTPUT_SHAPE,  BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DECONV_2D_H_
