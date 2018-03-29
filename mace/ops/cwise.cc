//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/cwise.h"

namespace mace {
namespace ops {

void Register_CWise(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("CWise")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    CWiseOp<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("CWise")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    CWiseOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("CWise")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    CWiseOp<DeviceType::OPENCL, half>);
}

}  // namespace ops
}  // namespace mace
