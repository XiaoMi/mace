//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/relu.h"
#include "mace/proto/mace.pb.h"
#if __ARM_NEON
#include "mace/kernels/neon/relu_neon.h"
#endif // __ARM_NEON

namespace mace {

REGISTER_CPU_OPERATOR(Relu, ReluOp<DeviceType::CPU, float>);

#if __ARM_NEON
template <>
bool ReluOp<DeviceType::NEON, float>::Run() {
  const Tensor* input_tensor = Input(0);
  Tensor* output_tensor = Output(0);
  kernels::NeonReluFuntion_float(input_tensor, output_tensor);
  return true;
}
REGISTER_NEON_OPERATOR(Relu, ReluOp<DeviceType::NEON, float>);
#endif // __ARM_NEON

} //  namespace mace
