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

#include "mace/core/allocator.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_allocator.h"
#endif

namespace mace {

std::map<int32_t, Allocator *> *gAllocatorRegistry() {
  static std::map<int32_t, Allocator *> g_allocator_registry;
  return &g_allocator_registry;
}

Allocator *GetDeviceAllocator(DeviceType type) {
  auto iter = gAllocatorRegistry()->find(type);
  if (iter == gAllocatorRegistry()->end()) {
    LOG(ERROR) << "Allocator not found for device " << type;
    return nullptr;
  }
  return iter->second;
}

MACE_REGISTER_ALLOCATOR(DeviceType::CPU, new CPUAllocator());
#ifdef MACE_ENABLE_OPENCL
MACE_REGISTER_ALLOCATOR(DeviceType::GPU, new OpenCLAllocator());
#endif
MACE_REGISTER_ALLOCATOR(DeviceType::HEXAGON, new CPUAllocator());

}  // namespace mace
