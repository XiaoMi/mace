//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_WINOGRAD_INVERSE_TRANSFORM_H_
#define MACE_OPS_WINOGRAD_INVERSE_TRANSFORM_H_

#include <memory>
#include <string>

#include "mace/core/operator.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/winograd_transform.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class WinogradInverseTransformOp : public Operator<D, T> {
 public:
  WinogradInverseTransformOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(OperatorBase::GetSingleArgument<int>("batch", 1),
                 OperatorBase::GetSingleArgument<int>("height", 0),
                 OperatorBase::GetSingleArgument<int>("width", 0),
                 kernels::StringToActivationType(
                     OperatorBase::GetSingleArgument<std::string>("activation",
                                                                  "NOOP")),
                 OperatorBase::GetSingleArgument<float>("max_limit", 0.0f)) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(INPUT);
    const Tensor *bias = this->InputSize() == 2 ? this->Input(BIAS) : nullptr;
    Tensor *output_tensor = this->Output(OUTPUT);
    functor_(input_tensor, bias, output_tensor, future);
    return true;
  }

 private:
  kernels::WinogradInverseTransformFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, BIAS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_WINOGRAD_INVERSE_TRANSFORM_H_
