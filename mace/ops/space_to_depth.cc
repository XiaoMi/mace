//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/space_to_depth.h"

namespace mace {
namespace ops {

void Register_SpaceToDepth(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("SpaceToDepth")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    SpaceToDepthOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("SpaceToDepth")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    SpaceToDepthOp<DeviceType::OPENCL, float>);
                    
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("SpaceToDepth")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    SpaceToDepthOp<DeviceType::OPENCL, half>);

}

}  // namespace ops
}  // namespace mace
