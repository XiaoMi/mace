//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/batch_to_space.h"

namespace mace {
namespace ops {

void Register_BatchToSpaceND(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("BatchToSpaceND")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    BatchToSpaceNDOp<DeviceType::OPENCL, float>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("BatchToSpaceND")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    BatchToSpaceNDOp<DeviceType::OPENCL, half>);
}

}  // namespace ops
}  // namespace mace
