//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/conv_2d.h"

namespace mace {

void Register_Conv2D(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Conv2D")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    Conv2dOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Conv2D")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    Conv2dOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Conv2D")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    Conv2dOp<DeviceType::OPENCL, half>);
}

}  // namespace mace
