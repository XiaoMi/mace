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
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ChannelShuffle")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ChannelShuffleOp<DeviceType::OPENCL, float>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ChannelShuffle")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    ChannelShuffleOp<DeviceType::OPENCL, half>);
}

}  // namespace mace
