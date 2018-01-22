//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_WINOGRAD_INVERSE_TRANSFORM_H_
#define MACE_OPS_WINOGRAD_INVERSE_TRANSFORM_H_

#include <memory>

#include "mace/core/operator.h"
#include "mace/kernels/winograd_transform.h"

namespace mace {

template<DeviceType D, typename T>
class WinogradInverseTransformOp : public Operator<D, T> {
 public:
  WinogradInverseTransformOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(OperatorBase::GetSingleArgument<int>("batch", 1),
                 OperatorBase::GetSingleArgument<int>("height", 0),
                 OperatorBase::GetSingleArgument<int>("width", 0)) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(INPUT);
    Tensor *output_tensor = this->Output(OUTPUT);

    functor_(input_tensor, output_tensor, future);
    return true;
  }

 private:
  kernels::WinogradInverseTransformFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_WINOGRAD_INVERSE_TRANSFORM_H_
