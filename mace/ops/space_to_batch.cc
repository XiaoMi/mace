//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/space_to_batch.h"

namespace mace {

void Register_SpaceToBatchND(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("SpaceToBatchND")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    SpaceToBatchNDOp<DeviceType::OPENCL, float>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("SpaceToBatchND")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    SpaceToBatchNDOp<DeviceType::OPENCL, half>);
}

}  //  namespace mace
