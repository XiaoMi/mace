//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CONV_POOL_2D_BASE_H_
#define MACE_OPS_CONV_POOL_2D_BASE_H_

#include "mace/core/operator.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {

template<DeviceType D, class T>
class ConvPool2dOpBase : public Operator<D, T> {
 public:
  ConvPool2dOpBase(const OperatorDef& op_def, Workspace* ws)
    : Operator<D, T>(op_def, ws),
    strides_(OperatorBase::GetRepeatedArgument<int>("strides")),
    padding_(static_cast<Padding>(
          OperatorBase::GetSingleArgument<int>("padding",
                                               static_cast<int>(SAME)))),
    dilations_(OperatorBase::GetRepeatedArgument<int>("dilations")) {}

 protected:
  std::vector<int> strides_;
  Padding padding_;
  std::vector<int> dilations_;
};

} // namespace mace

#endif // MACE_OPS_CONV_POOL_2D_BASE_H_
