//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/folded_batch_norm.h"

namespace mace {

void Register_FoldedBatchNorm(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry,
                    OpKeyBuilder("FoldedBatchNorm")
                        .Device(DeviceType::CPU)
                        .TypeConstraint<float>("T")
                        .Build(),
                    FoldedBatchNormOp<DeviceType::CPU, float>);

#if MACE_ENABLE_NEON
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("FoldedBatchNorm")
                                     .Device(DeviceType::NEON)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    FoldedBatchNormOp<DeviceType::NEON, float>);
#endif  // MACE_ENABLE_NEON

  REGISTER_OPERATOR(op_registry,
                    OpKeyBuilder("FoldedBatchNorm")
                        .Device(DeviceType::OPENCL)
                        .TypeConstraint<float>("T")
                        .Build(),
                    FoldedBatchNormOp<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry,
                    OpKeyBuilder("FoldedBatchNorm")
                        .Device(DeviceType::OPENCL)
                        .TypeConstraint<half>("T")
                        .Build(),
                    FoldedBatchNormOp<DeviceType::OPENCL, half>);
}

}  //  namespace mace
