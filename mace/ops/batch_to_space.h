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

#ifndef MACE_OPS_BATCH_TO_SPACE_H_
#define MACE_OPS_BATCH_TO_SPACE_H_

#include <memory>
#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/space_to_batch.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class BatchToSpaceNDOp : public Operator<D, T> {
 public:
  BatchToSpaceNDOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(OperatorBase::GetRepeatedArgs<int>("crops", {0, 0, 0, 0}),
                 OperatorBase::GetRepeatedArgs<int>("block_shape", {1, 1}),
                 true) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *batch_tensor = this->Input(INPUT);
    Tensor *space_tensor = this->Output(OUTPUT);
    return functor_(space_tensor, const_cast<Tensor *>(batch_tensor), future);
  }

 private:
  kernels::SpaceToBatchFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_BATCH_TO_SPACE_H_
