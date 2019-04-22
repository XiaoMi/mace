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

#include "mace/core/net_optimizer.h"

#include <string>

namespace mace {

DeviceType NetOptimizer::SelectBestDevice(
    const OperatorDef *op_def,
    DeviceType target_device_type,
    const std::set<DeviceType> &available_devices,
    const std::vector<DeviceType> &inputs_op_devices) {
  static const std::set<std::string> kComputeIntensiveOps = {
      "Conv2D", "DepthwiseConv2d", "Deconv2D", "DepthwiseDeconv2d",
      "FullyConnected"
  };
  // CPU is the device to fall back
  DeviceType best_device = DeviceType::CPU;
  if (available_devices.count(target_device_type) == 1) {
    best_device = target_device_type;
  }
  if (best_device == DeviceType::CPU) {
    return best_device;
  }
  // Put compute-intensive ops in target device
  if (kComputeIntensiveOps.count(op_def->type()) == 1) {
    return best_device;
  }
  // Greedy strategy: Use input op's device type as current op's device
  for (auto device_type : inputs_op_devices) {
    if (device_type != best_device) {
      best_device = device_type;
    }
  }
  return best_device;
}
}  // namespace mace
