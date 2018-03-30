//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CWISE_H_
#define MACE_OPS_CWISE_H_

#include <string>

#include "mace/core/operator.h"
#include "mace/kernels/cwise.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class CWiseOp : public Operator<D, T> {
 public:
  CWiseOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        x_(OperatorBase::GetSingleArgument<float>("x", 1.0)),
        functor_(static_cast<kernels::CWiseType>(
                     OperatorBase::GetSingleArgument<int>(
                         "type", static_cast<int>(
                             kernels::CWiseType::ADD))),
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
  kernels::CWiseFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CWISE_H_
