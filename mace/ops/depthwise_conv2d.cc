//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/depthwise_conv2d.h"

namespace mace {

void Register_DepthwiseConv2d(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("DepthwiseConv2d")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    DepthwiseConv2dOp<DeviceType::CPU, float>);

#if MACE_ENABLE_NEON
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("DepthwiseConv2d")
                                     .Device(DeviceType::NEON)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    DepthwiseConv2dOp<DeviceType::NEON, float>);
#endif  // MACE_ENABLE_NEON

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("DepthwiseConv2d")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    DepthwiseConv2dOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("DepthwiseConv2d")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    DepthwiseConv2dOp<DeviceType::OPENCL, half>);
}

}  // namespace mace
