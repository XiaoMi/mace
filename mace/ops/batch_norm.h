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

#ifndef MACE_OPS_BATCH_NORM_H_
#define MACE_OPS_BATCH_NORM_H_

#include "mace/core/operator.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/batch_norm.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class BatchNormOp : public Operator<D, T> {
 public:
  BatchNormOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(false, kernels::ActivationType::NOOP, 0.0f) {
    epsilon_ = OperatorBase::GetOptionalArg<float>("epsilon",
                                                   static_cast<float>(1e-4));
  }

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *scale = this->Input(SCALE);
    const Tensor *offset = this->Input(OFFSET);
    const Tensor *mean = this->Input(MEAN);
    const Tensor *var = this->Input(VAR);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional. ",
               input->dim_size());
    MACE_CHECK(scale->dim_size() == 1, "scale must be 1-dimensional. ",
               scale->dim_size());
    MACE_CHECK(offset->dim_size() == 1, "offset must be 1-dimensional. ",
               offset->dim_size());
    MACE_CHECK(mean->dim_size() == 1, "mean must be 1-dimensional. ",
               mean->dim_size());
    MACE_CHECK(var->dim_size() == 1, "var must be 1-dimensional. ",
               var->dim_size());

    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    return functor_(input, scale, offset, mean, var, epsilon_, output, future);
  }

 private:
  float epsilon_;
  kernels::BatchNormFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT, SCALE, OFFSET, MEAN, VAR);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_BATCH_NORM_H_
