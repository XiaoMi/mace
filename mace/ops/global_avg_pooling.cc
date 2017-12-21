//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/global_avg_pooling.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("GlobalAvgPooling")
                             .TypeConstraint<float>("T")
                             .Build(),
                      GlobalAvgPoolingOp<DeviceType::CPU, float>);

#if MACE_ENABLE_NEON
REGISTER_NEON_OPERATOR(OpKeyBuilder("GlobalAvgPooling")
                             .TypeConstraint<float>("T")
                             .Build(),
                       GlobalAvgPoolingOp<DeviceType::NEON, float>);
#endif  // MACE_ENABLE_NEON

}  //  namespace mace
