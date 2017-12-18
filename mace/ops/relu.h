//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_RELU_H_
#define MACE_OPS_RELU_H_

#include "mace/core/operator.h"
#include "mace/kernels/relu.h"

namespace mace {

template <DeviceType D, class T>
class ReluOp : public Operator<D, T> {
 public:
  ReluOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws) {
    functor_.max_limit_ =
        OperatorBase::GetSingleArgument<float>("max_limit", static_cast<float>(-1));
  }
  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->inputs_[0];
    Tensor *output_tensor = this->outputs_[0];
    output_tensor->ResizeLike(input_tensor);

    functor_(input_tensor, output_tensor, future);
    return true;
  }

 private:
  kernels::ReluFunctor<D, T> functor_;
};

}  // namespace mace

#endif  // MACE_OPS_RELU_H_
