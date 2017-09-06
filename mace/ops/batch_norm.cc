//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/batch_norm.h"

namespace mace {

REGISTER_CPU_OPERATOR(BatchNorm, BatchNormOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(BatchNorm, BatchNormOp<DeviceType::NEON, float>);
#endif // __ARM_NEON

} //  namespace mace