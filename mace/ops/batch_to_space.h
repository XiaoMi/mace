//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_SPACE_TO_BATCH_H_
#define MACE_OPS_SPACE_TO_BATCH_H_

#include <memory>

#include "mace/core/operator.h"
#include "mace/kernels/space_to_batch.h"

namespace mace {

template<DeviceType D, typename T>
class BatchToSpaceNDOp : public Operator<D, T> {
 public:
  BatchToSpaceNDOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(
            OperatorBase::GetRepeatedArgument<int>("crops", {0, 0, 0, 0}),
            OperatorBase::GetRepeatedArgument<int>("block_shape", {1, 1}),
            true) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);

    std::vector<index_t> output_shape(4, 0);
    BatchToSpaceHelper(input_tensor, output, output_shape);
    functor_(output, output_shape, const_cast<Tensor *>(input_tensor), future);
    return true;
  }

 private:
  inline void BatchToSpaceHelper(const Tensor *input_tensor,
                                 Tensor *output,
                                 std::vector<index_t> &output_shape) {
    auto crops = OperatorBase::GetRepeatedArgument<int>("crops", {0, 0, 0, 0});
    auto block_shape = OperatorBase::GetRepeatedArgument<int>("block_shape", {1, 1});
    MACE_CHECK(input_tensor->dim_size() == 4, "Input's shape should be 4D");
    MACE_CHECK(block_shape.size() == 2, "Block's shape should be 1D");
    MACE_CHECK(crops.size() == 4, "Crops' shape should be 2D");

    const index_t block_dims = block_shape.size();
    index_t block_shape_product = 1;
    for (uint32_t block_dim = 0; block_dim < block_dims; ++block_dim) {
      MACE_CHECK(block_shape[block_dim] > 1, "block_shape's value should be great to 1");
      const index_t block_shape_value = block_shape[block_dim];
      const index_t cropped_input_size = input_tensor->dim(block_dim + 1) * block_shape_value
          - crops[block_dim * 2]
          - crops[block_dim * 2 + 1];
      MACE_CHECK(cropped_input_size >= 0,
                 "cropped size must be non-negative");
      block_shape_product *= block_shape_value;
      output_shape[block_dim + 1] = cropped_input_size;
    }
    output_shape[0] = input_tensor->dim(0) / block_shape_product;
    output_shape[3] = input_tensor->dim(3);
  }

 private:
  kernels::SpaceToBatchFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_SPACE_TO_BATCH_H_
