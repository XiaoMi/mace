//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_NEG_H_
#define MACE_OPS_NEG_H_

#include <string>

#include "mace/core/operator.h"
#include "mace/kernels/negative.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class NegOp : public Operator<D, T> {
 public:
  NegOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_() {}

  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(0);    
    Tensor *output_tensor = this->outputs_[0];
    output_tensor->ResizeLike(input_tensor);

    functor_(input_tensor, output_tensor, future);
    return true;
  }

 private:
  kernels::NegFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_NEGATIVE_H_
