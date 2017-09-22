//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/conv_2d.h"

namespace mace {

REGISTER_CPU_OPERATOR(Conv2D, Conv2dOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(Conv2D, Conv2dOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

}  // namespace mace
