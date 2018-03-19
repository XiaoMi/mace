//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_RESHAPE_H_
#define MACE_OPS_RESHAPE_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/reshape.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class ReshapeOp : public Operator<D, T> {
 public:
  ReshapeOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        shape_(OperatorBase::GetRepeatedArgument<int64_t>("shape")) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const index_t num_dims = shape_.size();
    int unknown_idx = -1;
    index_t product = 1;
    std::vector<index_t> out_shape;

    for (int i = 0; i < num_dims; ++i) {
      if (shape_[i] == -1) {
        MACE_CHECK(unknown_idx == -1) << "Only one input size may be -1";
        unknown_idx = i;
        out_shape.push_back(1);
      } else {
        MACE_CHECK(shape_[i] >= 0) << "Shape must be non-negative: "
                                   << shape_[i];
        out_shape.push_back(shape_[i]);
        product *= shape_[i];
      }
    }

    if (unknown_idx != -1) {
      MACE_CHECK(product != 0)
          << "Cannot infer shape if there is zero shape size.";
      const index_t missing = input->size() / product;
      MACE_CHECK(missing * product == input->size())
          << "Input size not match reshaped tensor size";
      out_shape[unknown_idx] = missing;
    }

    Tensor *output = this->Output(OUTPUT);

    functor_(input, out_shape, output, future);
    return true;
  }

 private:
  std::vector<int64_t> shape_;
  kernels::ReshapeFunctor<D, T> functor_;

 private:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_RESHAPE_H_
