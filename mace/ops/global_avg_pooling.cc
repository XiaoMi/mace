//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/global_avg_pooling.h"

namespace mace {

void Register_GlobalAvgPooling(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("GlobalAvgPooling")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    GlobalAvgPoolingOp<DeviceType::CPU, float>);

#if MACE_ENABLE_NEON
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("GlobalAvgPooling")
                                     .Device(DeviceType::NEON)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    GlobalAvgPoolingOp<DeviceType::NEON, float>);
#endif  // MACE_ENABLE_NEON
}

}  //  namespace mace
