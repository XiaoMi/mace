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
  if (type == DeviceType::CPU) {
    return cpu_allocator();
  } else {
    REQUIRE(false, "device type ", type, " is not supported.");
  }
  return nullptr;
}

} // namespace mace
