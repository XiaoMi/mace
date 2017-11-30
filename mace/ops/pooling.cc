//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/pooling.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("Pooling")
                             .TypeConstraint<float>("T")
                             .Build(),
                      PoolingOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(OpKeyBuilder("Pooling")
                             .TypeConstraint<float>("T")
                             .Build(),
                       PoolingOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("Pooling")
                             .TypeConstraint<float>("T")
                             .Build(),
                         PoolingOp<DeviceType::OPENCL, float>);

}  //  namespace mace
