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

#ifndef MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DEVICE_H_
#define MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DEVICE_H_

#include <memory>
#include <utility>

#include "mace/core/device.h"
#include "mace/core/runtime/hexagon/hexagon_control_wrapper.h"
#ifdef MACE_ENABLE_HEXAGON
#include "mace/core/runtime/hexagon/hexagon_dsp_wrapper.h"
#endif
#ifdef MACE_ENABLE_HTA
#include "mace/core/runtime/hexagon/hexagon_hta_wrapper.h"
#endif

namespace mace {

class HexagonDevice : public CPUDevice {
 public:
  explicit HexagonDevice(DeviceType device_type,
                         utils::ThreadPool *thread_pool)
      : CPUDevice(0, AFFINITY_NONE, thread_pool),
        device_type_(device_type) {}

  DeviceType device_type() const override {
    return device_type_;
  };

 private:
  DeviceType device_type_;
};

std::unique_ptr<HexagonControlWrapper> CreateHexagonControlWrapper(
    Device *device) {
  std::unique_ptr<HexagonControlWrapper> hexagon_controller;
  auto device_type = device->device_type();
  switch (device_type) {
#ifdef MACE_ENABLE_HEXAGON
    case HEXAGON:
      hexagon_controller = make_unique<HexagonDSPWrapper>();
      break;
#endif
#ifdef MACE_ENABLE_HTA
    case HTA:
      hexagon_controller = make_unique<HexagonHTAWrapper>(device);
      break;
#endif
    default:LOG(FATAL) << "Not supported Hexagon device type: " << device_type;
  }

  return hexagon_controller;
}

}  // namespace mace
#endif  // MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DEVICE_H_
