//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_DEPTHWISE_CONV2D_H_
#define MACE_OPS_DEPTHWISE_CONV2D_H_

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
      : ConvPool2dOpBase<D, T>(op_def, ws),
        functor_(this->strides_.data(),
                 this->padding_,
                 this->dilations_.data(),
                 kernels::ActivationType::NOOP,
                 0.0f,
                 0.0f) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *filter = this->Input(FILTER);
    const Tensor *bias = nullptr;
    if (this->InputSize() >= 3) {
      bias = this->Input(BIAS);
    }
    Tensor *output = this->Output(OUTPUT);
    functor_(input, filter, bias, output, future);
    return true;
  }

 private:
  kernels::DepthwiseConv2dFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_DEPTHWISE_CONV2D_H_
