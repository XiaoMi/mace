//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/resize_bilinear.h"

namespace mace {

void Register_ResizeBilinear(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ResizeBilinear")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ResizeBilinearOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ResizeBilinear")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ResizeBilinearOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ResizeBilinear")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    ResizeBilinearOp<DeviceType::OPENCL, half>);
}

}  //  namespace mace
