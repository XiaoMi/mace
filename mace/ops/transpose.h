//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_TRANSPOSE_H_
#define MACE_OPS_TRANSPOSE_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/transpose.h"
#include "mace/kernels/softmax.h"

namespace mace {

template<DeviceType D, class T>
class TransposeOp : public Operator<D, T> {
 public:
  TransposeOp(const OperatorDef &operator_def, Workspace *ws)
    : Operator<D, T>(operator_def, ws),
      dims_(OperatorBase::GetRepeatedArgument<int>(
        "dims")),
      functor_(dims_) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    const std::vector<index_t> &input_shape = input->shape();
    MACE_CHECK(input_shape.size() == 4 && dims_.size() == 4,
               "rank should be 4");
    std::vector<index_t> output_shape;
    for (int i = 0; i < dims_.size(); ++i) {
      output_shape.push_back(input_shape[dims_[i]]);
    }
    output->Resize(output_shape);
    functor_(input, output, future);
    return true;
  }

 protected:
  std::vector<int> dims_;
  kernels::TransposeFunctor<D, T> functor_;

  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_TRANSPOSE_H_
