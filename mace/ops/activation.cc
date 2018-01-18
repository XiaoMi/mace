//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/activation.h"

namespace mace {

void Register_Activation(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Activation")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ActivationOp<DeviceType::CPU, float>);

#if MACE_ENABLE_NEON
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Activation")
                                     .Device(DeviceType::NEON)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ActivationOp<DeviceType::NEON, float>);
#endif  // MACE_ENABLE_NEON

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Activation")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ActivationOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Activation")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    ActivationOp<DeviceType::OPENCL, half>);
}

}  //  namespace mace
