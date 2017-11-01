//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_DEPTHWISE_CONV_H_
#define MACE_OPS_DEPTHWISE_CONV_H_

#include <memory>

#include "mace/core/operator.h"
#include "mace/kernels/conv_2d.h"
#include "mace/kernels/depthwise_conv2d.h"
#include "mace/ops/conv_pool_2d_base.h"

namespace mace {

template <DeviceType D, typename T>
class DepthwiseConv2dOp : public ConvPool2dOpBase<D, T> {
 public:
  DepthwiseConv2dOp(const OperatorDef &op_def, Workspace *ws)
      : ConvPool2dOpBase<D, T>(op_def, ws) {
    functor_.strides_ = this->strides_.data();
    functor_.dilations_ = this->dilations_.data();
  }

  bool Run() override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = nullptr;
    if (this->InputSize() >= 3) {
      bias = this->Input(BIAS);
    }
    Tensor *output = this->Output(OUTPUT);

    // resize filter shape.
    std::vector<index_t> filter_shape(filter->shape().begin(),
                                      filter->shape().end());
    filter_shape[0] *= filter_shape[1];
    filter_shape[1] = 1;
    std::vector<index_t> output_shape(4);
    std::vector<int> paddings(2);
    kernels::CalcPaddingAndOutputSize(
        input->shape().data(), filter_shape.data(), this->dilations_.data(),
        this->strides_.data(), this->padding_, output_shape.data(),
        paddings.data());
    output->Resize(output_shape);
    functor_.paddings_ = paddings;

    functor_(input, filter, bias, output);

    return true;
  }

 private:
  kernels::DepthwiseConv2dFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  //  namespace mace

#endif  //  MACE_OPS_DEPTHWISE_CONV_H_
