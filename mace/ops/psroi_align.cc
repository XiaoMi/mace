//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/psroi_align.h"

namespace mace {
namespace ops {

void Register_PSROIAlign(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("PSROIAlign")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    PSROIAlignOp<DeviceType::CPU, float>);
}

}  // namespace ops
}  // namespace mace
