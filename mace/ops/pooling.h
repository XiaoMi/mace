//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_POOLING_H_
#define MACE_OPS_POOLING_H_

#include "mace/core/operator.h"
#include "mace/kernels/pooling.h"
#include "mace/ops/conv_pool_2d_base.h"

namespace mace {

template <DeviceType D, class T>
class PoolingOp : public ConvPool2dOpBase<D, T> {
 public:
  PoolingOp(const OperatorDef &op_def, Workspace *ws)
      : ConvPool2dOpBase<D, T>(op_def, ws),
        kernels_(OperatorBase::GetRepeatedArgument<int>("kernels")),
        pooling_type_(
            static_cast<PoolingType>(OperatorBase::GetSingleArgument<int>(
                "pooling_type", static_cast<int>(AVG)))),
        functor_(pooling_type_,
                 kernels_.data(),
                 this->strides_.data(),
                 this->padding_type_,
                 this->paddings_,
                 this->dilations_.data()){};

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);

    functor_(input, output, future);
    return true;
  };

 protected:
  std::vector<int> kernels_;
  PoolingType pooling_type_;
  kernels::PoolingFunctor<D, T> functor_;

  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_POOLING_H_
