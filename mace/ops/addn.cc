//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/addn.h"

namespace mace {

REGISTER_CPU_OPERATOR(AddN, AddNOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(AddN, AddNOp<DeviceType::NEON, float>);
#endif // __ARM_NEON

} //  namespace mace
