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

#ifndef MACE_OPS_BIAS_ADD_H_
#define MACE_OPS_BIAS_ADD_H_

#include "mace/core/operator.h"
#include "mace/kernels/bias_add.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class BiasAddOp : public Operator<D, T> {
 public:
  BiasAddOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(static_cast<DataFormat>(OperatorBase::GetOptionalArg<int>(
            "data_format", NHWC))) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *bias = this->Input(BIAS);

    MACE_CHECK(bias->dim_size() == 1, "bias must be 1-dimensional. ",
               bias->dim_size());

    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    return functor_(input, bias, output, future);
  }

 private:
  kernels::BiasAddFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_BIAS_ADD_H_
