//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/relu.h"

namespace mace {

void Register_Relu(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Relu")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ReluOp<DeviceType::CPU, float>);

#if MACE_ENABLE_NEON
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Relu")
                                     .Device(DeviceType::NEON)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ReluOp<DeviceType::NEON, float>);
#endif  // MACE_ENABLE_NEON

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Relu")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ReluOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Relu")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    ReluOp<DeviceType::OPENCL, half>);
}

}  //  namespace mace
