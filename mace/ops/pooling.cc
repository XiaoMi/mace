//
// Copyright (c) 2017 XiaoMi All rights reserved.
//


#include "mace/ops/pooling.h"
#include "mace/proto/mace.pb.h"
#include "mace/kernels/pooling.h"

namespace mace {

REGISTER_CPU_OPERATOR(Pooling, PoolingOp<DeviceType::CPU, float>);

} //  namespace mace
