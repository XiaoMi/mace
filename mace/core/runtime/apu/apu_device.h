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

#ifndef MACE_CORE_RUNTIME_APU_APU_DEVICE_H_
#define MACE_CORE_RUNTIME_APU_APU_DEVICE_H_

#include "mace/core/device.h"

namespace mace {

class ApuDevice : public CPUDevice {
 public:
  explicit ApuDevice(utils::ThreadPool *thread_pool)
      : CPUDevice(0, AFFINITY_NONE, thread_pool) {}

  DeviceType device_type() const override {
    return DeviceType::APU;
  };
};

}  // namespace mace
#endif  // MACE_CORE_RUNTIME_APU_APU_DEVICE_H_
