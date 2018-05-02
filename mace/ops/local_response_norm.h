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

#ifndef MACE_OPS_LOCAL_RESPONSE_NORM_H_
#define MACE_OPS_LOCAL_RESPONSE_NORM_H_

#include "mace/core/operator.h"
#include "mace/kernels/local_response_norm.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class LocalResponseNormOp : public Operator<D, T> {
 public:
  LocalResponseNormOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_() {
    depth_radius_ = OperatorBase::GetSingleArgument<int>("depth_radius", 5);
    bias_ = OperatorBase::GetSingleArgument<float>("bias", 1.0f);
    alpha_ = OperatorBase::GetSingleArgument<float>("alpha", 1.0f);
    beta_ = OperatorBase::GetSingleArgument<float>("beta", 0.5f);
  }

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional. ",
               input->dim_size());

    Tensor *output = this->Output(OUTPUT);
    output->ResizeLike(input);

    functor_(input, depth_radius_, bias_, alpha_, beta_, output, future);
    return true;
  }

 private:
  int depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
  kernels::LocalResponseNormFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_LOCAL_RESPONSE_NORM_H_
