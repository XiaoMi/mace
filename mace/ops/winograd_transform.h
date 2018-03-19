//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_WINOGRAD_TRANSFORM_H_
#define MACE_OPS_WINOGRAD_TRANSFORM_H_

#include <memory>

#include "mace/core/operator.h"
#include "mace/kernels/winograd_transform.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class WinogradTransformOp : public Operator<D, T> {
 public:
  WinogradTransformOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(static_cast<Padding>(OperatorBase::GetSingleArgument<int>(
                     "padding", static_cast<int>(VALID))),
                 OperatorBase::GetRepeatedArgument<int>("padding_values")) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(INPUT);
    Tensor *output_tensor = this->Output(OUTPUT);

    functor_(input_tensor, output_tensor, future);
    return true;
  }

 private:
  kernels::WinogradTransformFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_WINOGRAD_TRANSFORM_H_
