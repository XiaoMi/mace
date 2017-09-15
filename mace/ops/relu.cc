//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/relu.h"

namespace mace {

REGISTER_CPU_OPERATOR(Relu, ReluOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(Relu, ReluOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

}  //  namespace mace
