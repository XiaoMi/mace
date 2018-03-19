//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_SOFTMAX_H_
#define MACE_OPS_SOFTMAX_H_

#include "mace/core/operator.h"
#include "mace/kernels/softmax.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class SoftmaxOp : public Operator<D, T> {
 public:
  SoftmaxOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws) {}

  bool Run(StatsFuture *future) override {
    const Tensor *logits = this->Input(LOGITS);

    Tensor *output = this->Output(OUTPUT);
    output->ResizeLike(logits);

    functor_(logits, output, future);
    return true;
  }

 private:
  kernels::SoftmaxFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(LOGITS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_SOFTMAX_H_
