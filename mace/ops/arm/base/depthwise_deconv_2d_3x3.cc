// Copyright 2020 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/ops/arm/base/depthwise_deconv_2d_3x3.h"

namespace mace {
namespace ops {
namespace arm {

void RegisterDepthwiseDeconv2dK3x3Delegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, DepthwiseDeconv2dK3x3S1<float>,
      delegator::DepthwiseDeconv2dParam,
      MACE_DELEGATOR_KEY_EX(DepthwiseDeconv2d, DeviceType::CPU,
                            float, ImplType::NEON, K3x3S1));
  MACE_REGISTER_DELEGATOR(
      registry, DepthwiseDeconv2dK3x3S2<float>,
      delegator::DepthwiseDeconv2dParam,
      MACE_DELEGATOR_KEY_EX(DepthwiseDeconv2d, DeviceType::CPU,
                            float, ImplType::NEON, K3x3S2));
}

void RegisterGroupDeconv2dK3x3Delegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, GroupDeconv2dK3x3S1<float>, delegator::GroupDeconv2dParam,
      MACE_DELEGATOR_KEY_EX(GroupDeconv2d, DeviceType::CPU,
                            float, ImplType::NEON, K3x3S1));
  MACE_REGISTER_DELEGATOR(
      registry, GroupDeconv2dK3x3S2<float>, delegator::GroupDeconv2dParam,
      MACE_DELEGATOR_KEY_EX(GroupDeconv2d, DeviceType::CPU,
                            float, ImplType::NEON, K3x3S2));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
