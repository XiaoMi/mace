//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_FOLDED_BATCH_NORM_H_
#define MACE_OPS_FOLDED_BATCH_NORM_H_

#include "mace/core/operator.h"
#include "mace/kernels/batch_norm.h"

namespace mace {

template <DeviceType D, class T>
class FoldedBatchNormOp : public Operator<D, T> {
 public:
  FoldedBatchNormOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(true,
                 kernels::StringToActivationType(
                     OperatorBase::GetSingleArgument<std::string>("activation",
                                                                  "NOOP")),
                 OperatorBase::GetSingleArgument<float>("max_limit", 0.0f),
                 OperatorBase::GetSingleArgument<float>("alpha", 0.0f)) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *scale = this->Input(SCALE);
    const Tensor *offset = this->Input(OFFSET);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional. ",
               input->dim_size());
    MACE_CHECK(scale->dim_size() == 1, "scale must be 1-dimensional. ",
               scale->dim_size());
    MACE_CHECK(offset->dim_size() == 1, "offset must be 1-dimensional. ",
               offset->dim_size());

    Tensor *output = this->Output(OUTPUT);
    output->ResizeLike(input);

    functor_(input, scale, offset, nullptr, nullptr, 0, output, future);
    return true;
  }

 private:
  kernels::BatchNormFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, SCALE, OFFSET, MEAN, VAR);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_FOLDED_BATCH_NORM_H_
