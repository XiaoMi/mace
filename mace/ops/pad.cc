//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/pad.h"

namespace mace {
namespace ops {

void Register_Pad(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Pad")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    PadOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Pad")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    PadOp<DeviceType::OPENCL, float>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Pad")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    PadOp<DeviceType::OPENCL, half>);
}

}  // namespace ops
}  // namespace mace
