//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/channel_shuffle.h"

namespace mace {

void Register_ChannelShuffle(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ChannelShuffle")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ChannelShuffleOp<DeviceType::CPU, float>);
}

}  // namespace mace
