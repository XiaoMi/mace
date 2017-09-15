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
  ReluOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<D, T>(operator_def, ws) {}
  bool Run() override {
    const Tensor* input_tensor = this->inputs_[0];
    Tensor* output_tensor = this->outputs_[0];
    output_tensor->ResizeLike(input_tensor);
    const T* input = input_tensor->data<T>();
    T* output = output_tensor->mutable_data<T>();
    index_t size = input_tensor->size();

    functor_(input, output, size);
    return true;
  }

 private:
  kernels::ReluFunctor<D, T> functor_;
};

}  // namespace mace

#endif  // MACE_OPS_RELU_H_
