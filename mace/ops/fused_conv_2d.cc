//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/fused_conv_2d.h"

namespace mace {
namespace ops {

void Register_FusedConv2D(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("FusedConv2D")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    FusedConv2dOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("FusedConv2D")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    FusedConv2dOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("FusedConv2D")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    FusedConv2dOp<DeviceType::OPENCL, half>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("FusedConv2D")
                                     .Device(DeviceType::NEON)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    FusedConv2dOp<DeviceType::NEON, float>);
}

}  // namespace ops
}  // namespace mace
