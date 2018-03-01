//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/bias_add.h"

namespace mace {

void Register_BiasAdd(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("BiasAdd")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    BiasAddOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("BiasAdd")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    BiasAddOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("BiasAdd")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    BiasAddOp<DeviceType::OPENCL, half>);
}

}  // namespace mace
