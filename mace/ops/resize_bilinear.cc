//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/resize_bilinear.h"

namespace mace {

REGISTER_CPU_OPERATOR(ResizeBilinear, ResizeBilinearOp<DeviceType::CPU, float>);

#if __ARM_NEON
REGISTER_NEON_OPERATOR(ResizeBilinear,
                       ResizeBilinearOp<DeviceType::NEON, float>);
#endif  // __ARM_NEON

}  //  namespace mace
