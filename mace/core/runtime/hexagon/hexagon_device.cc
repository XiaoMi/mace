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

#include "mace/core/runtime/hexagon/hexagon_device.h"

namespace mace {

HexagonDevice::HexagonDevice(DeviceType device_type,
                             utils::ThreadPool *thread_pool
#ifdef MACE_ENABLE_OPENCL
                             , std::unique_ptr<GPUDevice> gpu_device
#endif  // MACE_ENABLE_OPENCL
                             )
      : CPUDevice(0, AFFINITY_NONE, thread_pool),
        allocator_(make_unique<HexagonAllocator>()),
        device_type_(device_type)
#ifdef MACE_ENABLE_OPENCL
        , gpu_device_(std::move(gpu_device))
#endif  // MACE_ENABLE_OPENCL
         {}

#ifdef MACE_ENABLE_OPENCL
GPURuntime *HexagonDevice::gpu_runtime() {
  return gpu_device_->gpu_runtime();
}
#endif  // MACE_ENABLE_OPENCL

Allocator *HexagonDevice::allocator() {
#ifdef MACE_ENABLE_OPENCL
  return gpu_device_->allocator();
#else
  return allocator_.get();
#endif  // MACE_ENABLE_OPENCL
}

DeviceType HexagonDevice::device_type() const {
  return device_type_;
}

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
