//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/concat.h"

namespace mace {

REGISTER_CPU_OPERATOR(Concat, ConcatOp<DeviceType::CPU, float>);

}  // namespace mace
