//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_POOLING_H_
#define MACE_OPS_POOLING_H_

#include "mace/core/operator.h"
#include "mace/kernels/pooling.h"
#include "mace/ops/conv_pool_2d_base.h"

namespace mace {

template<DeviceType D, class T>
class PoolingOp : public ConvPool2dOpBase<D, T> {
public:
  PoolingOp(const OperatorDef& op_def, Workspace* ws)
  : ConvPool2dOpBase<D, T>(op_def, ws),
    kernels_(OperatorBase::GetRepeatedArgument<int>("kernels")),
    pooling_type_(static_cast<PoolingType>(
                  OperatorBase::GetSingleArgument<int>(
                  "pooling_type", static_cast<int>(AVG)))) {};

  bool Run() override{
    const Tensor* input = this->Input(INPUT);
    Tensor* output = this->Output(OUTPUT);
    std::vector<index_t> in_shape = input->shape();

    std::vector<index_t> output_shape;
    std::vector<int> paddings;
    std::vector<index_t> filter_shape = std::vector<index_t>(4);
    filter_shape[0] = in_shape[1];
    filter_shape[1] = in_shape[0];
    filter_shape[2] = kernels_[0];
    filter_shape[3] = kernels_[1];
    kernels::CalcPaddingAndOutputSize(in_shape.data(),
                                      filter_shape.data(),
                                      this->dilations_.data(),
                                      this->strides_.data(),
                                      this->padding_,
                                      &output_shape,
                                      &paddings);
    output->Resize(output_shape);

    auto pooling_func = kernels::PoolingFunctor<D, T>(pooling_type_,
                                                      kernels_.data(),
                                                      this->strides_.data(),
                                                      paddings.data(),
                                                      this->dilations_.data());
    pooling_func(input->data<float>(),
                 in_shape.data(),
                 output->mutable_data<float>(),
                 output->shape().data());
    return true;
  };

protected:
  std::vector<int> kernels_;
  PoolingType pooling_type_;

  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

} // namespace mace

#endif //MACE_OPS_POOLING_H_
