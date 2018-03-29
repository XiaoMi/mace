//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/transpose.h"

namespace mace {
namespace ops {

void Register_Transpose(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Transpose")
    .Device(DeviceType::CPU)
    .TypeConstraint<float>("T")
    .Build(),
                    TransposeOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Transpose")
    .Device(DeviceType::NEON)
    .TypeConstraint<float>("T")
    .Build(),
                    TransposeOp<DeviceType::NEON, float>);
}

}  // namespace ops
}  // namespace mace
