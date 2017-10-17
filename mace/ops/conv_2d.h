//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CONV_2D_H_
#define MACE_OPS_CONV_2D_H_

#include <memory>

#include "mace/core/operator.h"
#include "mace/kernels/conv_2d.h"
#include "mace/ops/conv_pool_2d_base.h"

namespace mace {

template <DeviceType D, typename T>
class Conv2dOp : public ConvPool2dOpBase<D, T> {
 public:
  Conv2dOp(const OperatorDef &op_def, Workspace *ws)
      : ConvPool2dOpBase<D, T>(op_def, ws) {
    functor_.strides_ = this->strides_.data();
    functor_.dilations_ = this->dilations_.data();
  }

  bool Run() override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const T *bias_data = nullptr;
    if (this->InputSize() >= 3) {
      const Tensor *bias = this->Input(BIAS);
      bias_data = bias->data<T>();
    }

    Tensor *output = this->Output(OUTPUT);

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    kernels::CalcPaddingAndOutputSize(
        input->shape().data(), filter->shape().data(), this->dilations_.data(),
        this->strides_.data(), this->padding_, output_shape.data(),
        paddings.data());
    output->Resize(output_shape);
    functor_.paddings_ = paddings;

    functor_(input->data<T>(), input->shape().data(), filter->data<T>(),
             filter->shape().data(), bias_data, output->mutable_data<T>(),
             output->shape().data());

    return true;
  }

 private:
  kernels::Conv2dFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_CONV_2D_H_
