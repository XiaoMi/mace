//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_SCALAR_MATH_H_
#define MACE_OPS_SCALAR_MATH_H_

#include <string>

#include "mace/core/operator.h"
#include "mace/kernels/scalar_math.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class ScalarMathOp : public Operator<D, T> {
 public:
  ScalarMathOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        x_(OperatorBase::GetSingleArgument<float>("x", 1.0)),
        functor_(static_cast<kernels::ScalarMathType>(
                     OperatorBase::GetSingleArgument<int>(
                         "type", static_cast<int>(
                             kernels::ScalarMathType::ADD))),
                 this->x_) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(INPUT);
    Tensor *output_tensor = this->Output(OUTPUT);
    output_tensor->ResizeLike(input_tensor);

    functor_(input_tensor, output_tensor, future);
    return true;
  }

 protected:
  const float x_;
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);

 private:
  kernels::ScalarMathFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SCALAR_MATH_H_
