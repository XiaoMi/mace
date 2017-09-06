//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_RELU_H_
#define MACE_OPS_RELU_H_

#include "mace/core/operator.h"
#include "mace/kernels/relu.h"

namespace mace {

template<DeviceType D, class T>
class ReluOp : public Operator<D, T> {
 public:
  ReluOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws) {}
  bool Run() override {
    const Tensor* input_tensor = this->Input(0);
    Tensor* output_tensor = this->Output(0);
    kernels::ReluFuntion<T>(input_tensor, output_tensor);
    return true;
  }
};

} // namespace mace

#endif // MACE_OPS_RELU_H_
