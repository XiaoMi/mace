//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/relu.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("Relu")
                          .TypeConstraint<float>("T")
                          .Build(),
                      ReluOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(OpKeyBuilder("Relu")
                             .TypeConstraint<float>("T")
                             .Build(),
                       ReluOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("Relu")
                             .TypeConstraint<float>("T")
                             .Build(),
                         ReluOp<DeviceType::OPENCL, float>);

}  //  namespace mace
