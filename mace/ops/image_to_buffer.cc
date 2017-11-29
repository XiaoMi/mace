//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/image_to_buffer.h"

namespace mace {

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("ImageToBuffer")
                             .TypeConstraint<float>("T")
                             .Build(),
                         ImageToBufferOp<DeviceType::OPENCL, float>);

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("ImageToBuffer")
                             .TypeConstraint<half>("T")
                             .Build(),
                         ImageToBufferOp<DeviceType::OPENCL, half>);

}  //  namespace mace
