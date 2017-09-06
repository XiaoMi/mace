//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CONV_2D_H_
#define MACE_OPS_CONV_2D_H_

#include "mace/core/operator.h"

namespace mace {

template<DeviceType D, class T>
class Conv2dOp : public Operator<D, T> {
 public:
  Conv2dOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        kernels_(OperatorBase::GetRepeatedArgument<int>("kernels")),
        strides_(OperatorBase::GetRepeatedArgument<int>("strides")),
        paddings_(OperatorBase::GetRepeatedArgument<int>("paddings")),
        dilations_(OperatorBase::GetRepeatedArgument<int>("dilations")) {}

  bool Run() override;

 private:
  vector<int> kernels_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;

  OP_INPUT_TAGS(INPUT, FILTER, BIAS);
  OP_OUTPUT_TAGS(OUTPUT);
};

} // namespace mace

#endif // MACE_OPS_CONV_2D_H_
