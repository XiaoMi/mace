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

} // namespace mace
