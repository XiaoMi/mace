//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_BATCH_NORM_H_
#define MACE_BATCH_NORM_H_

#include "mace/core/operator.h"
#include "mace/kernels/batch_norm.h"

namespace mace {

template <DeviceType D, class T>
class BatchNormOp : public Operator<D, T> {
 public:
  BatchNormOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws), functor_() {}

  bool Run() override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *scale = this->Input(SCALE);
    const Tensor *offset = this->Input(OFFSET);
    const Tensor *mean = this->Input(MEAN);
    const Tensor *var = this->Input(VAR);
    const Tensor *epsilon = this->Input(EPSILON);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional. ",
               input->dim_size());
    MACE_CHECK(scale->dim_size() == 1, "scale must be 1-dimensional. ",
               scale->dim_size());
    MACE_CHECK(offset->dim_size() == 1, "offset must be 1-dimensional. ",
               offset->dim_size());
    MACE_CHECK(mean->dim_size() == 1, "mean must be 1-dimensional. ",
               mean->dim_size());
    MACE_CHECK(var->dim_size() == 1, "var must be 1-dimensional. ",
               var->dim_size());
    MACE_CHECK(epsilon->dim_size() == 0, "epsilon must be 0-dimensional. ",
               epsilon->dim_size());

    Tensor *output = this->Output(OUTPUT);
    output->ResizeLike(input);

    functor_(input, scale, offset, mean, var, epsilon, output);
    return true;
  }

 private:
  kernels::BatchNormFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, SCALE, OFFSET, MEAN, VAR, EPSILON);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  //  namespace mace

#endif  //  MACE_BATCH_NORM_H_
