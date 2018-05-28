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

#ifndef MACE_OPS_SPACE_TO_DEPTH_H_
#define MACE_OPS_SPACE_TO_DEPTH_H_

#include <memory>
#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/depth_to_space.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class SpaceToDepthOp : public Operator<D, T> {
 public:
  SpaceToDepthOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(OperatorBase::GetOptionalArg<int>("block_size", 1), false) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    MACE_CHECK(input->dim_size() == 4, "input dim should be 4");
    const int block_size = OperatorBase::GetOptionalArg<int>("block_size", 1);
    index_t input_height;
    index_t input_width;
    index_t input_depth;
    if (D == CPU) {
      input_height = input->dim(2);
      input_width = input->dim(3);
      input_depth = input->dim(1);
    } else if (D == GPU) {
      input_height = input->dim(1);
      input_width = input->dim(2);
      input_depth = input->dim(3);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    MACE_CHECK((input_depth % 4) == 0,
               "input channel should be dividable by 4");
    MACE_CHECK(
        (input_width % block_size == 0) && (input_height % block_size == 0),
        "input width and height should be dividable by block_size",
        input->dim(3));
    return functor_(input, output, future);
  }

 protected:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);

 private:
  kernels::DepthToSpaceOpFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SPACE_TO_DEPTH_H_
