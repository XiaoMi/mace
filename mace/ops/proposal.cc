//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/proposal.h"

namespace mace {
namespace ops {

void Register_Proposal(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("Proposal")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    ProposalOp<DeviceType::CPU, float>);
}

}  // namespace ops
}  // namespace mace
