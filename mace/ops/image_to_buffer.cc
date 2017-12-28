//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/image_to_buffer.h"

namespace mace {

void Register_ImageToBuffer(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ImageToBuffer")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ImageToBufferOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ImageToBuffer")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    ImageToBufferOp<DeviceType::OPENCL, half>);
}

}  //  namespace mace
