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
}

}  // namespace mace
