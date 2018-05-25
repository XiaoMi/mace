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

#ifndef MACE_OPS_PSROI_ALIGN_H_
#define MACE_OPS_PSROI_ALIGN_H_

#include "mace/core/operator.h"
#include "mace/kernels/psroi_align.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class PSROIAlignOp : public Operator<D, T> {
 public:
  PSROIAlignOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(OperatorBase::GetSingleArgument<T>("spatial_scale", 0),
                 OperatorBase::GetSingleArgument<int>("output_dim", 0),
                 OperatorBase::GetSingleArgument<int>("group_size", 0)) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *rois = this->Input(ROIS);

    Tensor *output = this->Output(OUTPUT);

    return functor_(input, rois, output, future);
  }

 private:
  kernels::PSROIAlignFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, ROIS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_PSROI_ALIGN_H_
