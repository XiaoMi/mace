//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_ELTWISE_H_
#define MACE_OPS_ELTWISE_H_

#include "mace/core/operator.h"
#include "mace/kernels/eltwise.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class EltwiseOp : public Operator<D, T> {
 public:
  EltwiseOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(static_cast<kernels::EltwiseType>(
                     OperatorBase::GetSingleArgument<int>(
                         "type", static_cast<int>(kernels::EltwiseType::SUM))),
                 OperatorBase::GetRepeatedArgument<float>("coeff")) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input0 = this->Input(0);
    const Tensor *input1 = this->Input(1);
    Tensor *output = this->Output(OUTPUT);
    MACE_CHECK(input0->dim_size() == input1->dim_size())
        << "Inputs of Eltwise op must be same shape";
    for (int i = 0; i < input0->dim_size(); ++i) {
      MACE_CHECK(input0->dim(i) == input1->dim(i))
          << "Inputs of Eltwise op must be same shape";
    }

    output->ResizeLike(input0);

    functor_(input0, input1, output, future);
    return true;
  }

 private:
  kernels::EltwiseFunctor<D, T> functor_;

 private:
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ELTWISE_H_
