//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_PAD_H_
#define MACE_OPS_PAD_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/pad.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class PadOp : public Operator<D, T> {
 public:
  PadOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(OperatorBase::GetRepeatedArgument<int>("paddings"),
                 OperatorBase::GetSingleArgument<float>("constant_value", 0.0))
  {}

  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(0);
    Tensor *output_tensor = this->Output(0);
    functor_(input_tensor, output_tensor, future);
    return true;
  }

 private:
  kernels::PadFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_PAD_H_
