//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/allocator.h"

namespace mace {

static std::unique_ptr<CPUAllocator> g_cpu_allocator(new CPUAllocator());
CPUAllocator* cpu_allocator() {
  return g_cpu_allocator.get();
}

void SetCPUAllocator(CPUAllocator* alloc) {
  g_cpu_allocator.reset(alloc);
}

Allocator* GetDeviceAllocator(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
    case DeviceType::NEON:
      return cpu_allocator();
    default:
      MACE_CHECK(false, "device type ", type, " is not supported.");
  }
  return nullptr;
}

} // namespace mace
