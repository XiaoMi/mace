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

#include "mace/ops/arm/base/deconv_2d_4x4.h"

namespace mace {
namespace ops {
namespace arm {

void RegisterDeconv2dK4x4Delegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Deconv2dK4x4S1<float>, delegator::Deconv2dParam,
      MACE_DELEGATOR_KEY_EX(Deconv2d, DeviceType::CPU,
                            float, ImplType::NEON, K4x4S1));
  MACE_REGISTER_DELEGATOR(
      registry, Deconv2dK4x4S2<float>, delegator::Deconv2dParam,
      MACE_DELEGATOR_KEY_EX(Deconv2d, DeviceType::CPU,
                            float, ImplType::NEON, K4x4S2));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
