//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/depth_to_space.h"

namespace mace {
namespace ops {

void Register_DepthToSpace(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("DepthToSpace")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    DepthToSpaceOp<DeviceType::CPU, float>);
/*
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("DepthToSpace")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    DepthToSpaceOp<DeviceType::OPENCL, float>);
                    
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("DepthToSpace")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    DepthToSpaceOp<DeviceType::OPENCL, half>);
*/
}

}  // namespace ops
}  // namespace mace
