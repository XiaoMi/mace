//
// Copyright (c) 2017 XiaoMi All rights reserved.
//


#include "mace/ops/pooling.h"
#include "mace/proto/mace.pb.h"
#include "mace/kernels/pooling.h"

namespace mace {

template <>
bool PoolingOp<DeviceType::CPU, float>::Run() {
  const Tensor* input_tensor = Input(0);
  Tensor* output_tensor = Output(0);
  int pooling_type = this->GetSingleArgument<int>("pooling_type", 0);
  int kernel_size = this->GetSingleArgument<int>("kernel_size", 1);
  int stride = this->GetSingleArgument<int>("stride", 1);
  int padding = this->GetSingleArgument<int>("padding", 0);
  kernels::PoolingFunction<float>(input_tensor, output_tensor, pooling_type, kernel_size, stride, padding);
  return true;
}
REGISTER_CPU_OPERATOR(Pooling, PoolingOp<DeviceType::CPU, float>);

} //  namespace mace
