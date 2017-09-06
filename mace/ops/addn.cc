//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/addn.h"
#include "mace/proto/mace.pb.h"
#if __ARM_NEON
#include "mace/kernels/neon/addn_neon.h"
#endif // __ARM_NEON

namespace mace {

REGISTER_CPU_OPERATOR(AddN, AddNOp<DeviceType::CPU, float>);

#if __ARM_NEON
template <>
bool AddNOp<DeviceType::NEON, float>::Run() {
  Tensor* output_tensor = Output(0);
  kernels::NeonAddNFuntion_float(Inputs(), output_tensor);
  return true;
}
REGISTER_NEON_OPERATOR(AddN, AddNOp<DeviceType::NEON, float>);
#endif // __ARM_NEON

} //  namespace mace
