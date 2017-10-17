//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/depthwise_conv2d.h"

namespace mace {

REGISTER_CPU_OPERATOR(DepthwiseConv2d,
                      DepthwiseConv2dOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(DepthwiseConv2d,
                       DepthwiseConv2dOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

}  // namespace mace
