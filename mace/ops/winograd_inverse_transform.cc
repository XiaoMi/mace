//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/winograd_inverse_transform.h"

namespace mace {

void Register_WinogradInverseTransform(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("WinogradInverseTransform")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    WinogradInverseTransformOp<DeviceType::OPENCL, float>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("WinogradInverseTransform")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    WinogradInverseTransformOp<DeviceType::OPENCL, half>);
}

}  //  namespace mace
