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

template<DeviceType D, typename T>
class Conv2dOp : public ConvPool2dOpBase<D, T> {
 public:
  Conv2dOp(const OperatorDef& op_def, Workspace* ws)
    : ConvPool2dOpBase<D, T>(op_def, ws) {};

  bool Run() override {
    const Tensor* input = this->Input(INPUT);
    const Tensor* filter = this->Input(FILTER);
    const Tensor* bias = this->Input(BIAS);
    Tensor* output = this->Output(OUTPUT);

    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    kernels::CalcPaddingAndOutputSize(input->shape().data(),
                                      filter->shape().data(),
                                      this->dilations_.data(),
                                      this->strides_.data(),
                                      this->padding_,
                                      output_shape.data(),
                                      paddings.data());
    output->Resize(output_shape);

    auto conv2d = kernels::Conv2dFunctor<D, T>(this->strides_.data(),
                                               paddings.data(),
                                               this->dilations_.data());
    conv2d(input->data<T>(), input->shape().data(),
           filter->data<T>(), filter->shape().data(),
           bias->data<T>(), output->mutable_data<T>(),
           output->shape().data());

    return true;
  }

 protected:
  OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  OP_OUTPUT_TAGS(OUTPUT);
};

} // namespace mace

#endif // MACE_OPS_CONV_2D_H_
