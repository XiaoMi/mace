//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/buffer_to_image.h"

namespace mace {

void Register_BufferToImage(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("BufferToImage")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    BufferToImageOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("BufferToImage")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    BufferToImageOp<DeviceType::OPENCL, half>);
}

}  // namespace mace
