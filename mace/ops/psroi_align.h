//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_PSROI_ALIGN_H_
#define MACE_OPS_PSROI_ALIGN_H_

#include "mace/core/operator.h"
#include "mace/kernels/psroi_align.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class PSROIAlignOp : public Operator<D, T> {
 public:
  PSROIAlignOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(OperatorBase::GetSingleArgument<T>("spatial_scale", 0),
                 OperatorBase::GetSingleArgument<int>("output_dim", 0),
                 OperatorBase::GetSingleArgument<int>("group_size", 0)) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *rois = this->Input(ROIS);

    Tensor *output = this->Output(OUTPUT);

    functor_(input, rois, output, future);
    return true;
  }

 private:
  kernels::PSROIAlignFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, ROIS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_PSROI_ALIGN_H_
