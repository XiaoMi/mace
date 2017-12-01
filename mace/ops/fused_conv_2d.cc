//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/fused_conv_2d.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("FusedConv2D")
                             .TypeConstraint<float>("T")
                             .Build(),
                      FusedConv2dOp<DeviceType::CPU, float>);

REGISTER_CPU_OPERATOR(OpKeyBuilder("FusedConv2D")
                          .TypeConstraint<half>("T")
                          .Build(),
                      FusedConv2dOp<DeviceType::CPU, half>);


REGISTER_OPENCL_OPERATOR(OpKeyBuilder("FusedConv2D")
                             .TypeConstraint<float>("T")
                             .Build(),
                         FusedConv2dOp<DeviceType::OPENCL, float>);

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("FusedConv2D")
                             .TypeConstraint<half>("T")
                             .Build(),
                         FusedConv2dOp<DeviceType::OPENCL, half>);

}  // namespace mace
