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

#ifndef MACE_OPS_POOLING_H_
#define MACE_OPS_POOLING_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/pooling.h"
#include "mace/ops/conv_pool_2d_base.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class PoolingOp : public ConvPool2dOpBase<D, T> {
 public:
  PoolingOp(const OperatorDef &op_def, Workspace *ws)
      : ConvPool2dOpBase<D, T>(op_def, ws),
        kernels_(OperatorBase::GetRepeatedArgs<int>("kernels")),
        pooling_type_(
            static_cast<PoolingType>(OperatorBase::GetOptionalArg<int>(
                "pooling_type", static_cast<int>(AVG)))),
        functor_(pooling_type_,
                 kernels_.data(),
                 this->strides_.data(),
                 this->padding_type_,
                 this->paddings_,
                 this->dilations_.data()) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);

    return functor_(input, output, future);
  };

 protected:
  std::vector<int> kernels_;
  PoolingType pooling_type_;
  kernels::PoolingFunctor<D, T> functor_;

  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_POOLING_H_
