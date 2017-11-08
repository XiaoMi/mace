//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_SPACE_TO_BATCH_H_
#define MACE_OPS_SPACE_TO_BATCH_H_

#include <memory>

#include "mace/core/operator.h"
#include "mace/kernels/space_to_batch.h"

namespace mace {

static void SpaceToBatchHelper(const Tensor *input_tensor,
                               const Tensor *block_shape_tensor,
                               const Tensor *paddings_tensor,
                               Tensor *output) {
  MACE_CHECK(input_tensor->dim_size() == 4, "Input's shape should be 4D");
  MACE_CHECK(block_shape_tensor->dim_size() == 1, "Block's shape should be 1D");
  MACE_CHECK(paddings_tensor->dim_size() == 2, "Paddings' shape should be 2D");

  const index_t block_dims = block_shape_tensor->dim(0);
  MACE_CHECK(block_dims == paddings_tensor->dim(0) && 2 == paddings_tensor->dim(1));
  Tensor::MappingGuard block_shape_tensor_mapper(block_shape_tensor);
  Tensor::MappingGuard padding_tensor_mapper(paddings_tensor);
  const int *block_shape_ptr = block_shape_tensor->data<int>();
  const int *paddings_ptr = paddings_tensor->data<int>();
  std::vector<index_t> output_shape(4, 0);
  index_t block_shape_product = 1;
  for (uint32_t block_dim = 0; block_dim < block_dims; ++block_dim) {
    MACE_CHECK(block_shape_ptr[block_dim] > 1, "block_shape's value should be great to 1");
    const index_t block_shape_value = block_shape_ptr[block_dim];
    const index_t padded_input_size = input_tensor->dim(block_dim + 2)
                                      + *paddings_ptr
                                      + *(paddings_ptr+1);
    MACE_CHECK(padded_input_size % block_shape_value == 0,
               "padded input is not divisible by block_shape");
    block_shape_product *= block_shape_value;
    output_shape[block_dim+2] = padded_input_size / block_shape_value;
    paddings_ptr += 2;
  }
  output_shape[0] = input_tensor->dim(0) * block_shape_product;
  output_shape[1] = input_tensor->dim(1);

  output->Resize(output_shape);
}

template <DeviceType D, typename T>
class SpaceToBatchNDOp : public Operator<D, T> {
 public:
  SpaceToBatchNDOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws) {}

  bool Run() override {
    const Tensor *input_tensor = this->Input(INPUT);
    const Tensor *block_shape_tensor = this->Input(BLOCK_SHAPE);
    const Tensor *paddings_tensor = this->Input(PADDINGS);
    Tensor *output = this->Output(OUTPUT);

    SpaceToBatchHelper(input_tensor, block_shape_tensor, paddings_tensor, output);
    functor_(const_cast<Tensor*>(input_tensor), block_shape_tensor, paddings_tensor, output);
    return true;
  }

 private:
  kernels::SpaceToBatchFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, BLOCK_SHAPE, PADDINGS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_SPACE_TO_BATCH_H_
