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

#include "mace/ops/arm/base/conv_2d_5x5.h"

namespace mace {
namespace ops {
namespace arm {

void RegisterConv2dK5x5Delegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Conv2dK5x5S1<float>, delegator::Conv2dParam,
      MACE_DELEGATOR_KEY_EX(Conv2d, DeviceType::CPU,
                            float, ImplType::NEON, K5x5S1));
}

}  // namespace arm
}  // namespace ops
}  // namespace mace
