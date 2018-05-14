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

#ifndef MACE_OPS_SPACE_TO_BATCH_H_
#define MACE_OPS_SPACE_TO_BATCH_H_

#include <memory>
#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/space_to_batch.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class SpaceToBatchNDOp : public Operator<D, T> {
 public:
  SpaceToBatchNDOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(
            OperatorBase::GetRepeatedArgument<int>("paddings", {0, 0, 0, 0}),
            OperatorBase::GetRepeatedArgument<int>("block_shape", {1, 1}),
            false) {}

  bool Run(StatsFuture *future) override {
    const Tensor *space_tensor = this->Input(INPUT);
    Tensor *batch_tensor = this->Output(OUTPUT);

    std::vector<index_t> output_shape(4, 0);
    CalculateOutputShape(space_tensor, output_shape.data());
    functor_(const_cast<Tensor *>(space_tensor), output_shape, batch_tensor,
             future);
    return true;
  }

 private:
  inline void CalculateOutputShape(const Tensor *input_tensor,
                                   index_t *output_shape) {
    auto paddings =
        OperatorBase::GetRepeatedArgument<int>("paddings", {0, 0, 0, 0});
    auto block_shape =
        OperatorBase::GetRepeatedArgument<int>("block_shape", {1, 1});
    MACE_CHECK(input_tensor->dim_size() == 4, "Input's shape should be 4D");
    MACE_CHECK(block_shape.size() == 2, "Block's shape should be 1D");
    MACE_CHECK(paddings.size() == 4, "Paddings' shape should be 2D");

    const index_t block_dims = block_shape.size();
    index_t block_shape_product = 1;
    for (uint32_t block_dim = 0; block_dim < block_dims; ++block_dim) {
      MACE_CHECK(block_shape[block_dim] > 1,
                 "block_shape's value should be great to 1");
      const index_t block_shape_value = block_shape[block_dim];
      const index_t padded_input_size = input_tensor->dim(block_dim + 1) +
                                        paddings[block_dim * 2] +
                                        paddings[block_dim * 2 + 1];
      MACE_CHECK(padded_input_size % block_shape_value == 0, "padded input ",
                 padded_input_size, " is not divisible by block_shape");
      block_shape_product *= block_shape_value;
      output_shape[block_dim + 1] = padded_input_size / block_shape_value;
    }
    output_shape[0] = input_tensor->dim(0) * block_shape_product;
    output_shape[3] = input_tensor->dim(3);
  }

 private:
  kernels::SpaceToBatchFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SPACE_TO_BATCH_H_
