//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_ACTIVATION_H_
#define MACE_OPS_ACTIVATION_H_

#include "mace/core/operator.h"
#include "mace/kernels/activation.h"

namespace mace {

template <DeviceType D, class T>
class ActivationOp : public Operator<D, T> {
 public:
  ActivationOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(kernels::StringToActivationType(
                     OperatorBase::GetSingleArgument<std::string>("activation",
                                                                  "NOOP")),
                 OperatorBase::GetSingleArgument<float>("max_limit", 0.0f)) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(0);
    const Tensor *alpha_tensor = this->InputSize() >= 2 ? this->Input(1) : nullptr;
    Tensor *output_tensor = this->outputs_[0];
    output_tensor->ResizeLike(input_tensor);

    functor_(input_tensor, alpha_tensor, output_tensor, future);
    return true;
  }

 private:
  kernels::ActivationFunctor<D, T> functor_;
};

}  // namespace mace

#endif  // MACE_OPS_ACTIVATION_H_
