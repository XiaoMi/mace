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

#ifndef MACE_CORE_MEMORY_MEMORY_MANAGER_H_
#define MACE_CORE_MEMORY_MEMORY_MANAGER_H_

#include <set>
#include <string>

#include "mace/core/memory/allocator.h"

namespace mace {

struct MemInfo;
class Allocator;

class MemoryManager {
 public:
  explicit MemoryManager(Allocator *allocator) : allocator_(allocator) {}
  virtual ~MemoryManager() {}
  Allocator *GetAllocator() { return allocator_; }

  virtual void *ObtainMemory(const MemInfo &info,
                             const BufRentType rent_type) = 0;
  virtual void ReleaseMemory(void *ptr, const BufRentType rent_type) = 0;

  virtual std::vector<index_t> GetMemoryRealSize(const void *ptr) = 0;

  virtual void ReleaseAllMemory(const BufRentType rent_type, bool del_buf) = 0;

 protected:
  Allocator *allocator_;
};

}  // namespace mace



#endif  // MACE_CORE_MEMORY_MEMORY_MANAGER_H_
