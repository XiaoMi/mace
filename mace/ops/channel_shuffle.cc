//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/channel_shuffle.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("ChannelShuffle")
                             .TypeConstraint<float>("T")
                             .Build(),
                      ChannelShuffleOp<DeviceType::CPU, float>);

}  // namespace mace
