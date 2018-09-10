// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include "mace/core/device.h"

namespace mace {

CPUDevice::CPUDevice(const int num_threads)
    : cpu_runtime_(new CPURuntime(num_threads)) {}

CPUDevice::~CPUDevice() = default;

CPURuntime *CPUDevice::cpu_runtime() {
  return cpu_runtime_.get();
}

#ifdef MACE_ENABLE_OPENCL
OpenCLRuntime *CPUDevice::opencl_runtime() {
  return nullptr;
}
#endif

Allocator *CPUDevice::allocator() {
  return GetCPUAllocator();
}

DeviceType CPUDevice::device_type() const {
  return DeviceType::CPU;
}

}  // namespace mace
