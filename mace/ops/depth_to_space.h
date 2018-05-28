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

#ifndef MACE_OPS_DEPTH_TO_SPACE_H_
#define MACE_OPS_DEPTH_TO_SPACE_H_

#include <memory>
#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/depth_to_space.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class DepthToSpaceOp : public Operator<D, T> {
 public:
  DepthToSpaceOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        block_size_(OperatorBase::GetOptionalArg<int>("block_size", 1)),
        functor_(this->block_size_, true) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    MACE_CHECK(input->dim_size() == 4, "input dim should be 4");

    int input_depth;
    if (D == CPU) {
      input_depth = input->dim(1);
    } else if (D == GPU) {
      input_depth = input->dim(3);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    MACE_CHECK(input_depth % (block_size_ * block_size_) == 0,
               "input depth should be dividable by block_size * block_size",
               input_depth);
    MACE_CHECK((input_depth % 4) == 0,
               "input channel should be dividable by 4");
    return functor_(input, output, future);
  }

 protected:
  const int block_size_;
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);

 private:
  kernels::DepthToSpaceOpFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DEPTH_TO_SPACE_H_
