// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace test {

TEST(MaceDeviceCapalibityAPITest, Capability) {
  auto cpu_capability = GetCapability(DeviceType::CPU);
  LOG(INFO) << "CPU: float32 "
            << cpu_capability.float32_performance.exec_time
            << " vs quantized8 "
            << cpu_capability.quantized8_performance.exec_time;
  auto gpu_capability = GetCapability(
      DeviceType::GPU, cpu_capability.float32_performance.exec_time);
  if (gpu_capability.supported) {
    LOG(INFO) << "GPU: float32 "
              << gpu_capability.float32_performance.exec_time
              << " vs quantized8 "
              << gpu_capability.quantized8_performance.exec_time;
  }
}

}  // namespace test
}  // namespace mace
