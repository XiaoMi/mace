//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/reorganize.h"

namespace mace {
namespace ops {

void Register_ReOrganize(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ReOrganize")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ReOrganizeOp<DeviceType::CPU, float>);
}

}  // namespace ops
}  // namespace mace
