//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/batch_to_space.h"

namespace mace {

REGISTER_OPENCL_OPERATOR(BatchToSpaceND, BatchToSpaceNDOp<DeviceType::OPENCL, float>);

}  //  namespace mace
