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

#include "mace/core/memory/general_memory_manager.h"

#include <string>

#include "mace/core/memory/allocator.h"
#include "mace/utils/logging.h"

namespace mace {

GeneralMemoryManager::GeneralMemoryManager(Allocator *allocator)
    : MemoryManager(allocator) {}

GeneralMemoryManager::~GeneralMemoryManager() {}

void *GeneralMemoryManager::ObtainMemory(const MemInfo &info,
                                         const BufRentType rent_type) {
  if (shared_pools_.count(rent_type) == 0) {
    shared_pools_.emplace(rent_type, make_unique<MemoryPool>(allocator_));
  }
  return shared_pools_.at(rent_type)->ObtainMemory(info);
}

void GeneralMemoryManager::ReleaseMemory(void *ptr,
                                         const BufRentType rent_type) {
  if (shared_pools_.count(rent_type) == 0) {
    LOG(WARNING) << "There is no memory in the rent pool: " << rent_type;
    return;
  }
  shared_pools_.at(rent_type)->ReleaseMemory(ptr);
}

void GeneralMemoryManager::ReleaseAllMemory(const BufRentType rent_type,
                                            bool del_buf) {
  if (shared_pools_.count(rent_type) > 0) {
    shared_pools_.at(rent_type)->ReleaseAllMemory(del_buf);
  }
}

GeneralMemoryManager::MemoryPool::MemoryPool(Allocator *allocator)
    : allocator_(allocator) {}

GeneralMemoryManager::MemoryPool::~MemoryPool() {
  ClearMemory();
}

void GeneralMemoryManager::MemoryPool::ClearMemory() {
  for (BlockList::iterator iter = mem_used_blocks_.begin();
       iter != mem_used_blocks_.end(); ++iter) {
    VLOG(2) << "Finally release used memory, size: " << iter->first;
    allocator_->Delete(iter->second);
  }
  mem_used_blocks_.clear();

  for (BlockList::iterator iter = mem_free_blocks_.begin();
       iter != mem_free_blocks_.end(); ++iter) {
    VLOG(2) << "Finally release unused memory, size: " << iter->first;
    allocator_->Delete(iter->second);
  }
  mem_free_blocks_.clear();
}

void *GeneralMemoryManager::MemoryPool::ObtainMemory(const MemInfo &mem_info) {
  MACE_CHECK(mem_info.mem_type == allocator_->GetMemType());
  size_t bytes = mem_info.bytes();
  auto iter = mem_free_blocks_.lower_bound(bytes);
  void *ptr = nullptr;
  if (iter == mem_free_blocks_.end()) {
    MACE_CHECK_SUCCESS(allocator_->New(mem_info, &ptr));
    mem_used_blocks_.emplace(bytes, ptr);
    VLOG(2) << "GeneralMemoryManager::MemoryPool::ObtainMemory New memory: "
            << MakeString(mem_info.dims) << ", ptr = " << ptr;
  } else {
    ptr = iter->second;
    mem_used_blocks_.emplace(iter->first, iter->second);
    mem_free_blocks_.erase(iter);
    VLOG(2) << "GeneralMemoryManager::MemoryPool::ObtainMemory Old memory: "
            << MakeString(mem_info.dims) << ", ptr = " << ptr
            << ", mem type: " << static_cast<int>(mem_info.mem_type);
  }
  return ptr;
}

void GeneralMemoryManager::MemoryPool::ReleaseMemory(void *ptr) {
  // TODO(luxuhui): If the blocks become more, we can add a `map` to optimize
  // the follow code.
  for (BlockList::iterator iter = mem_used_blocks_.begin();
       iter != mem_used_blocks_.end(); ++iter) {
    if (iter->second == ptr) {
      mem_free_blocks_.emplace(iter->first, iter->second);
      VLOG(2) << "ReleaseMemory, ptr: " << ptr;
      iter = mem_used_blocks_.erase(iter);
      return;
    }
  }
  VLOG(1) << "ReleaseMemory, but find an unknown ptr: " << ptr;
}

void GeneralMemoryManager::MemoryPool::ReleaseAllMemory(bool del_buf) {
  if (del_buf) {
    ClearMemory();
  } else {
    for (BlockList::iterator iter = mem_used_blocks_.begin();
         iter != mem_used_blocks_.end(); iter = mem_used_blocks_.erase(iter)) {
      mem_free_blocks_.emplace(iter->first, iter->second);
      VLOG(2) << "ReleaseAllMemory, ptr: " << iter->second;
    }
    mem_used_blocks_.clear();
  }
}

}  // namespace mace


