//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/batch_norm.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("BatchNorm")
                             .TypeConstraint<float>("T")
                             .Build(),
                      BatchNormOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(OpKeyBuilder("BatchNorm")
                             .TypeConstraint<float>("T")
                             .Build(),
                       BatchNormOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("BatchNorm")
                             .TypeConstraint<float>("T")
                             .Build(),
                         BatchNormOp<DeviceType::OPENCL, float>);

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("BatchNorm")
                             .TypeConstraint<half>("T")
                             .Build(),
                         BatchNormOp<DeviceType::OPENCL, half>);

}  //  namespace mace
