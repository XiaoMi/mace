//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/reshape.h"

namespace mace {

void Register_Reshape(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Reshape")
      .Device(DeviceType::CPU)
      .TypeConstraint<float>("T")
      .Build(),
                    ReshapeOp<DeviceType::CPU, float>);
}

}  //  namespace mace
