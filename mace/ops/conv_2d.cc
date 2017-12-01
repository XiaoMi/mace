//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/conv_2d.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("Conv2D")
                             .TypeConstraint<float>("T")
                             .Build(),
                      Conv2dOp<DeviceType::CPU, float>);

REGISTER_CPU_OPERATOR(OpKeyBuilder("Conv2D")
                          .TypeConstraint<half>("T")
                          .Build(),
                      Conv2dOp<DeviceType::CPU, half>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(OpKeyBuilder("Conv2D")
                             .TypeConstraint<float>("T")
                             .Build(),
                       Conv2dOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("Conv2D")
                             .TypeConstraint<float>("T")
                             .Build(),
                         Conv2dOp<DeviceType::OPENCL, float>);

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("Conv2D")
                             .TypeConstraint<half>("T")
                             .Build(),
                         Conv2dOp<DeviceType::OPENCL, half>);

}  // namespace mace
