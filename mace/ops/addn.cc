//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/addn.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("AddN")
                             .TypeConstraint<float>("T")
                             .Build(),
                      AddNOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(OpKeyBuilder("AddN")
                             .TypeConstraint<float>("T")
                             .Build(),
                       AddNOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("AddN")
                             .TypeConstraint<float>("T")
                             .Build(),
                         AddNOp<DeviceType::OPENCL, float>);

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("AddN")
                             .TypeConstraint<half>("T")
                             .Build(),
                         AddNOp<DeviceType::OPENCL, half>);

}  //  namespace mace
