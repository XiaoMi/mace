//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/local_response_norm.h"

namespace mace {
namespace ops {

void Register_LocalResponseNorm(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("LocalResponseNorm")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    LocalResponseNormOp<DeviceType::CPU, float>);
}

}  // namespace ops
}  // namespace mace
