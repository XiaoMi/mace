//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_BATCH_NORM_H_
#define MACE_BATCH_NORM_H_

#include "mace/core/operator.h"
#include "mace/kernels/batch_norm.h"

namespace mace {

template<DeviceType D, class T>
class BatchNormOp : public Operator<D, T> {
  public:
    BatchNormOp(const OperatorDef &operator_def, Workspace *ws)
            : Operator<D, T>(operator_def, ws),
              functor_(OperatorBase::GetSingleArgument<float>("variance_epsilon", 1e-4)){}

    bool Run() override {
      const Tensor* input = this->Input(0);
      const Tensor* scale = this->Input(1);
      const Tensor* offset = this->Input(2);
      const Tensor* mean = this->Input(3);
      const Tensor* var = this->Input(4);

      MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional. ", input->dim_size());
      MACE_CHECK(scale->dim_size() == 1, "scale must be 1-dimensional. ", scale->dim_size());
      MACE_CHECK(offset->dim_size() == 1, "offset must be 1-dimensional. ", offset->dim_size());
      MACE_CHECK(mean->dim_size() == 1, "mean must be 1-dimensional. ", mean->dim_size());
      MACE_CHECK(var->dim_size() == 1, "var must be 1-dimensional. ", var->dim_size());

      Tensor* output = this->Output(0);
      output->ResizeLike(input);

      const TIndex n = input->dim(0);
      const TIndex channel = input->dim(1);
      const TIndex sample_size = input->dim(2) * input->dim(3);

      const float* input_ptr = input->data<float>();
      const float* scale_ptr = scale->data<float>();
      const float* offset_ptr = offset->data<float>();
      const float* mean_ptr = mean->data<float>();
      const float* var_ptr = var->data<float>();
      float* output_ptr = output->mutable_data<float>();

      functor_(input_ptr, scale_ptr, offset_ptr, mean_ptr, var_ptr,
                                     n, channel, sample_size,
                                     output_ptr);
      return true;
    }
  private:
    kernels::BatchNormFunctor<D, T> functor_;

};

} //  namespace mace

#endif //  MACE_BATCH_NORM_H_
