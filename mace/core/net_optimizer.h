//  Copyright 2019 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_CORE_NET_OPTIMIZER_H_
#define MACE_CORE_NET_OPTIMIZER_H_

#include <set>
#include <vector>

#include "mace/port/port.h"
#include "mace/proto/mace.pb.h"

namespace mace {

class NetOptimizer {
 public:
  DeviceType SelectBestDevice(const OperatorDef *op_def,
                              DeviceType target_device,
                              const std::set<DeviceType> &available_devices,
                              const std::vector<DeviceType> &inputs_op_devices);
};

}  // namespace mace
#endif  // MACE_CORE_NET_OPTIMIZER_H_
