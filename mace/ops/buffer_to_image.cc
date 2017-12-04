//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/buffer_to_image.h"

namespace mace {

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("BufferToImage")
                             .TypeConstraint<float>("T")
                             .Build(),
                         BufferToImageOp<DeviceType::OPENCL, float>);

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("BufferToImage")
                             .TypeConstraint<half>("T")
                             .Build(),
                         BufferToImageOp<DeviceType::OPENCL, half>);

}  //  namespace mace
