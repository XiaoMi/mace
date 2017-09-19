//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_DEPTHWISE_CONV_H_
#define MACE_OPS_DEPTHWISE_CONV_H_

#include <memory>

#include "mace/core/operator.h"
#include "mace/kernels/conv_2d.h"
#include "mace/ops/conv_pool_2d_base.h"
#include "mace/kernels/depthwise_conv2d.h"

namespace mace {

template <DeviceType D, typename T>
class DepthwiseConv2dOp : public ConvPool2dOpBase<D, T> {
 public:
  DepthwiseConv2dOp(const OperatorDef& op_def, Workspace* ws)
      : ConvPool2dOpBase<D, T>(op_def, ws),
        functor_(this->Input(INPUT)->shape().data(),
                 this->Input(FILTER)->shape().data(),
                 this->strides_.data(), this->padding_, this->dilations_.data()){};

  bool Run() override {
    const Tensor* input = this->Input(INPUT);
    const Tensor* filter = this->Input(FILTER);
    const Tensor* bias = this->Input(BIAS);
    Tensor* output = this->Output(OUTPUT);

    // resize filter shape.
    std::vector<index_t> filter_shape(filter->shape().begin(), filter->shape().end());
    filter_shape[0] *= filter_shape[1];
    filter_shape[1]  = 1;
    std::vector<index_t> output_shape(4);
    this->CalOutputSize(input->shape().data(), filter_shape.data(), output_shape.data());
    output->Resize(output_shape);

    functor_(input->data<T>(), input->shape().data(), filter->data<T>(),
             filter_shape.data(), bias->data<T>(), output->mutable_data<T>(),
             output->shape().data());

    return true;
  }

 private:
  kernels::DepthwiseConv2dFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  OP_OUTPUT_TAGS(OUTPUT);
};

} //  namespace mace

#endif //  MACE_OPS_DEPTHWISE_CONV_H_
