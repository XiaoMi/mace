//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/concat.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("Concat")
                             .TypeConstraint<float>("T")
                             .Build(),
                      ConcatOp<DeviceType::CPU, float>);

}  // namespace mace
