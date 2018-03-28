//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/scalar_math.h"

namespace mace {
namespace ops {

void Register_ScalarMath(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ScalarMath")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ScalarMathOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ScalarMath")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ScalarMathOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("ScalarMath")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    ScalarMathOp<DeviceType::OPENCL, half>);
}

}  // namespace ops
}  // namespace mace
