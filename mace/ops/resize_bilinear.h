//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_RESIZE_BILINEAR_H
#define MACE_RESIZE_BILINEAR_H

#include "mace/core/operator.h"
#include "mace/kernels/resize_bilinear.h"

namespace mace {

template <DeviceType D, class T>
class ResizeBilinearOp : public Operator<D, T> {
 public:
  ResizeBilinearOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(
            OperatorBase::GetSingleArgument<bool>("align_corners", false)) {}

  bool Run() override {
    const Tensor *input = this->Input(0);
    const Tensor *resize_dims = this->Input(1);
    Tensor *output = this->Output(0);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional.",
               input->dim_size());
    MACE_CHECK(resize_dims->dim_size() == 1,
               "resize dim must be 2-dimensional.", resize_dims->dim_size());

    functor_(input, resize_dims, output);
    return true;
  }

 private:
  kernels::ResizeBilinearFunctor<D, T> functor_;
};

}  // namespace mace

#endif  // MACE_RESIZE_BILINEAR_H
