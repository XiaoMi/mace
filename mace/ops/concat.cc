//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/concat.h"

namespace mace {

REGISTER_CPU_OPERATOR(OpKeyBuilder("Concat")
                          .TypeConstraint<float>("T")
                          .Build(),
                      ConcatOp<DeviceType::CPU, float>);
REGISTER_CPU_OPERATOR(OpKeyBuilder("Concat")
                          .TypeConstraint<half>("T")
                          .Build(),
                      ConcatOp<DeviceType::CPU, half>);

REGISTER_OPENCL_OPERATOR(OpKeyBuilder("Concat")
                             .TypeConstraint<float>("T")
                             .Build(),
                         ConcatOp<DeviceType::OPENCL, float>);
REGISTER_OPENCL_OPERATOR(OpKeyBuilder("Concat")
                             .TypeConstraint<half>("T")
                             .Build(),
                         ConcatOp<DeviceType::OPENCL, half>);
}  // namespace mace
