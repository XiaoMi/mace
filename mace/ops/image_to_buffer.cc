//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/image_to_buffer.h"

namespace mace {

REGISTER_OPENCL_OPERATOR(ImageToBuffer, ImageToBufferOp<DeviceType::OPENCL, float>);

}  //  namespace mace
