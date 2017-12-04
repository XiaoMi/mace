//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/pooling.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("Pooling")
                             .TypeConstraint<float>("T")
                             .Build(),
                      PoolingOp<DeviceType::CPU, float>);
REGISTER_CPU_OPERATOR(OpKeyBuilder("Pooling")
                          .TypeConstraint<half>("T")
                          .Build(),
                      PoolingOp<DeviceType::CPU, half>);

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
REGISTER_OPENCL_OPERATOR(OpKeyBuilder("Pooling")
                             .TypeConstraint<half>("T")
                             .Build(),
                         PoolingOp<DeviceType::OPENCL, half>);

}  //  namespace mace
