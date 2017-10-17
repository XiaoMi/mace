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
    const Tensor *input = this->Input(0);
    const Tensor *scale = this->Input(1);
    const Tensor *offset = this->Input(2);
    const Tensor *mean = this->Input(3);
    const Tensor *var = this->Input(4);
    const Tensor *epsilon = this->Input(5);

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

    Tensor *output = this->Output(0);
    output->ResizeLike(input);

    const index_t n = input->dim(0);
    const index_t channel = input->dim(1);
    const index_t sample_size = input->dim(2) * input->dim(3);

    const T *input_ptr = input->data<T>();
    const T *scale_ptr = scale->data<T>();
    const T *offset_ptr = offset->data<T>();
    const T *mean_ptr = mean->data<T>();
    const T *var_ptr = var->data<T>();
    const T *epsilon_ptr = epsilon->data<T>();
    T *output_ptr = output->mutable_data<T>();

    functor_(input_ptr, scale_ptr, offset_ptr, mean_ptr, var_ptr, *epsilon_ptr,
             n, channel, sample_size, output_ptr);
    return true;
  }

 private:
  kernels::BatchNormFunctor<D, T> functor_;
};

}  //  namespace mace

#endif  //  MACE_BATCH_NORM_H_
