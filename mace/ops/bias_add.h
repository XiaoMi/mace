//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_BIAS_ADD_H_
#define MACE_BIAS_ADD_H_

#include "mace/core/operator.h"
#include "mace/kernels/bias_add.h"

namespace mace {

template <DeviceType D, class T>
class BiasAddOp : public Operator<D, T> {
 public:
  BiasAddOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws), functor_() {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *bias = this->Input(BIAS);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional. ",
               input->dim_size());
    MACE_CHECK(bias->dim_size() == 1, "bias must be 1-dimensional. ",
               bias->dim_size());

    Tensor *output = this->Output(OUTPUT);
    output->ResizeLike(input);

    functor_(input, bias, output, future);
    return true;
  }

 private:
  kernels::BiasAddFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, BIAS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_BIAS_ADD_H_
