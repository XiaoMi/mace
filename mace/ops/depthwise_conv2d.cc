//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/depthwise_conv2d.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("DepthwiseConv2d")
                             .TypeConstraint<float>("T")
                             .Build(),
                      DepthwiseConv2dOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(OpKeyBuilder("DepthwiseConv2d")
                             .TypeConstraint<float>("T")
                             .Build(),
                       DepthwiseConv2dOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("DepthwiseConv2d")
                             .TypeConstraint<float>("T")
                             .Build(),
                         DepthwiseConv2dOp<DeviceType::OPENCL, float>);

}  // namespace mace
