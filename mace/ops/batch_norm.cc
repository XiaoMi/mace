//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/batch_norm.h"

namespace mace {

void Register_BatchNorm(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("BatchNorm")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    BatchNormOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("BatchNorm")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    BatchNormOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("BatchNorm")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    BatchNormOp<DeviceType::OPENCL, half>);
}

}  // namespace mace
