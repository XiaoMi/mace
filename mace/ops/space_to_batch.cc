//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/space_to_batch.h"

namespace mace {

REGISTER_OPENCL_OPERATOR(SpaceToBatchND, SpaceToBatchNDOp<DeviceType::OPENCL, float>);

}  //  namespace mace
