//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_SLICE_H_
#define MACE_OPS_SLICE_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/slice.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class SliceOp : public Operator<D, T> {
 public:
  SliceOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws) {}

  bool Run(StatsFuture *future) override {
    MACE_CHECK(this->OutputSize() >= 2) << "There must be at least two outputs for slicing";
    const Tensor *input = this->Input(INPUT);
    const std::vector<Tensor *> output_list = this->Outputs();
    MACE_CHECK((input->dim(3) % this->OutputSize()) == 0) << "Outputs do not split input equally.";

    functor_(input, output_list, future);
    return true;
  }

 private:
  kernels::SliceFunctor<D, T> functor_;

 private:
  OP_INPUT_TAGS(INPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SLICE_H_
