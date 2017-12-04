//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/bias_add.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("BiasAdd")
                             .TypeConstraint<float>("T")
                             .Build(),
                      BiasAddOp<DeviceType::CPU, float>);

/*
#if __ARM_NEON
REGISTER_NEON_OPERATOR(OpKeyBuilder("BiasAdd")
                             .TypeConstraint<float>("T")
                             .Build(),
                       BiasAddOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON
*/

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("BiasAdd")
                             .TypeConstraint<float>("T")
                             .Build(),
                         BiasAddOp<DeviceType::OPENCL, float>);

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("BiasAdd")
                             .TypeConstraint<half>("T")
                             .Build(),
                         BiasAddOp<DeviceType::OPENCL, half>);

}  //  namespace mace
