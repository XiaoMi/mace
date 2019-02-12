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

#include "mace/core/allocator.h"

#include <unistd.h>
#include <sys/mman.h>
#include <memory>

namespace mace {

Allocator *GetCPUAllocator() {
  static CPUAllocator allocator;
  return &allocator;
}

void AdviseFree(void *addr, size_t length) {
  int page_size = sysconf(_SC_PAGESIZE);
  void *addr_aligned =
      reinterpret_cast<void *>(
          (reinterpret_cast<uintptr_t>(addr) + page_size - 1)
              & (~(page_size - 1)));
  uintptr_t delta =
      reinterpret_cast<uintptr_t>(addr_aligned)
          - reinterpret_cast<uintptr_t>(addr);
  if (length >= delta + page_size) {
    size_t len_aligned = (length - delta) & (~(page_size - 1));
    int ret = madvise(addr_aligned, len_aligned, MADV_DONTNEED);
    if (ret != 0) {
      LOG(ERROR) << "Advise free failed: " << strerror(errno);
    }
  }
}

}  // namespace mace
