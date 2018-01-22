//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/winograd_transform.h"

namespace mace {

void Register_WinogradTransform(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("WinogradTransform")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    WinogradTransformOp<DeviceType::OPENCL, float>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("WinogradTransform")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    WinogradTransformOp<DeviceType::OPENCL, half>);
}

}  //  namespace mace
