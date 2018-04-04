//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/slice.h"

namespace mace {
namespace ops {

void Register_Slice(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Slice")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    SliceOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Slice")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    SliceOp<DeviceType::OPENCL, float>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Slice")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    SliceOp<DeviceType::OPENCL, half>);
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Slice")
                                     .Device(DeviceType::NEON)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    SliceOp<DeviceType::NEON, float>);
}

}  // namespace ops
}  // namespace mace
