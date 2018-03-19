//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_RESIZE_BILINEAR_H_
#define MACE_OPS_RESIZE_BILINEAR_H_

#include "mace/core/operator.h"
#include "mace/kernels/resize_bilinear.h"

namespace mace {

template <DeviceType D, class T>
class ResizeBilinearOp : public Operator<D, T> {
 public:
  ResizeBilinearOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(
            OperatorBase::GetRepeatedArgument<index_t>("size", {-1, -1}),
            OperatorBase::GetSingleArgument<bool>("align_corners", false)) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional.",
               input->dim_size());

    functor_(input, output, future);
    return true;
  }

 private:
  kernels::ResizeBilinearFunctor<D, T> functor_;
};

}  // namespace mace

#endif  // MACE_OPS_RESIZE_BILINEAR_H_
