//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

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
MACE_REGISTER_ALLOCATOR(DeviceType::NEON, new CPUAllocator());
#ifdef MACE_ENABLE_OPENCL
MACE_REGISTER_ALLOCATOR(DeviceType::OPENCL, new OpenCLAllocator());
#endif
MACE_REGISTER_ALLOCATOR(DeviceType::HEXAGON, new CPUAllocator());

}  // namespace mace
