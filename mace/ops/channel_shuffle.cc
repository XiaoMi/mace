//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/channel_shuffle.h"

namespace mace {

REGISTER_CPU_OPERATOR(ChannelShuffle, ChannelShuffleOp<DeviceType::CPU, float>);

}  // namespace mace
