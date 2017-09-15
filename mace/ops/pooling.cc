//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/pooling.h"

namespace mace {

REGISTER_CPU_OPERATOR(Pooling, PoolingOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(Pooling, PoolingOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

}  //  namespace mace
