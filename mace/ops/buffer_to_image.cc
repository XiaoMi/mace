//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/buffer_to_image.h"

namespace mace {

REGISTER_OPENCL_OPERATOR(BufferToImage, BufferToImageOp<DeviceType::OPENCL, float>);

}  //  namespace mace
