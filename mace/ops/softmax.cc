//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/softmax.h"

namespace mace {
namespace ops {

void Register_Softmax(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Softmax")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    SoftmaxOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Softmax")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    SoftmaxOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Softmax")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    SoftmaxOp<DeviceType::OPENCL, half>);
}

}  // namespace ops
}  // namespace mace
