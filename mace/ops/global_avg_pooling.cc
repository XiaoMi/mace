//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/global_avg_pooling.h"

namespace mace {

REGISTER_CPU_OPERATOR(GlobalAvgPooling,
                      GlobalAvgPoolingOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(GlobalAvgPooling,
                       GlobalAvgPoolingOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

}  //  namespace mace
