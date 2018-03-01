//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/pooling.h"

namespace mace {

void Register_Pooling(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Pooling")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    PoolingOp<DeviceType::CPU, float>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Pooling")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    PoolingOp<DeviceType::CPU, half>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Pooling")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    PoolingOp<DeviceType::OPENCL, float>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Pooling")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    PoolingOp<DeviceType::OPENCL, half>);
}

}  // namespace mace
