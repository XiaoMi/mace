//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/reduce_mean.h"

namespace mace {
namespace ops {

void Register_ReduceMean(OperatorRegistry *op_registry) {
  MACE_REGISTER_OPERATOR(op_registry, OpKeyBuilder("ReduceMean")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ReduceMeanOp<DeviceType::CPU, float>);
#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OPERATOR(op_registry, OpKeyBuilder("ReduceMean")
                                     .Device(DeviceType::GPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ReduceMeanOp<DeviceType::GPU, float>);

  MACE_REGISTER_OPERATOR(op_registry, OpKeyBuilder("ReduceMean")
                                     .Device(DeviceType::GPU)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    ReduceMeanOp<DeviceType::GPU, half>);
#endif
}

}  // namespace ops
}  // namespace mace
