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

#ifndef MACE_CORE_MEMORY_GENERAL_MEMORY_MANAGER_H_
#define MACE_CORE_MEMORY_GENERAL_MEMORY_MANAGER_H_

#include <string>

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "mace/core/memory/memory_manager.h"
#include "mace/core/types.h"

namespace mace {

class GeneralMemoryManager : public MemoryManager {
 public:
  explicit GeneralMemoryManager(Allocator *allocator);
  ~GeneralMemoryManager();

  void *ObtainMemory(const MemInfo &info, const BufRentType rent_type) override;
  void ReleaseMemory(void *ptr, const BufRentType rent_type) override;

  void ReleaseAllMemory(const BufRentType rent_type, bool del_buf) override;


  typedef std::multimap<index_t, void *> BlockList;
  class MemoryPool {
   public:
    MemoryPool() {}
    explicit MemoryPool(Allocator *allocator);
    ~MemoryPool();

    void *ObtainMemory(const MemInfo &info);
    void ReleaseMemory(void *ptr);
    void ReleaseAllMemory(bool del_buf);

   private:
    void ClearMemory();

   private:
    BlockList mem_used_blocks_;
    BlockList mem_free_blocks_;
    Allocator *allocator_;
  };

 private:
  // namespace and buffer pool
  typedef std::unordered_map<int, std::unique_ptr<MemoryPool>> SharedPools;
  SharedPools shared_pools_;
};

}  // namespace mace



#endif  // MACE_CORE_MEMORY_GENERAL_MEMORY_MANAGER_H_
