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

#include "mace/runtimes/opencl/opencl_image_manager.h"

#include <string>
#include "mace/utils/logging.h"

namespace mace {

OpenclImageManager::OpenclImageManager(Allocator *allocator)
    : MemoryManager(allocator) {}

OpenclImageManager::~OpenclImageManager() {}

void *OpenclImageManager::ObtainMemory(const MemInfo &info,
                                       const BufRentType rent_type) {
  if (shared_pools_.count(rent_type) == 0) {
    shared_pools_.emplace(rent_type, make_unique<ImagePool>(allocator_));
  }
  return shared_pools_.at(rent_type)->ObtainMemory(info);
}

void OpenclImageManager::ReleaseMemory(void *ptr, const BufRentType rent_type) {
  if (shared_pools_.count(rent_type) == 0) {
    LOG(FATAL) << "OpenclImageManager::ReleaseSharedMemory"
               << " in an unknown namespace: " << rent_type;
    return;
  }
  shared_pools_.at(rent_type)->ReleaseMemory(ptr);
}

void OpenclImageManager::ReleaseAllMemory(const BufRentType rent_type,
                                          bool del_buf) {
  if (shared_pools_.count(rent_type) > 0) {
    shared_pools_.at(rent_type)->ReleaseAllMemory(del_buf);
    return;
  }
}

OpenclImageManager::ImagePool::ImagePool(Allocator *allocator)
    : allocator_(allocator) {}

OpenclImageManager::ImagePool::~ImagePool() {
  ClearMemory();
}

void OpenclImageManager::ImagePool::ClearMemory() {
  for (BlockList::iterator iter = mem_used_blocks_.begin();
       iter != mem_used_blocks_.end(); ++iter) {
    VLOG(2) << "Finally release image, size: " << iter->first
            << ", width: " << iter->second->dims[0]
            << ", height: " << iter->second->dims[1];
    allocator_->Delete(iter->second->mutable_memory<void>());
  }
  mem_used_blocks_.clear();

  for (BlockList::iterator iter = mem_free_blocks_.begin();
       iter != mem_free_blocks_.end(); ++iter) {
    allocator_->Delete(iter->second->mutable_memory<void>());
  }
  mem_free_blocks_.clear();
}

void *OpenclImageManager::ImagePool::ObtainMemory(const MemInfo &info) {
  MACE_CHECK(info.mem_type == MemoryType::GPU_IMAGE);
  auto size = info.size();
  auto iter = mem_free_blocks_.lower_bound(size);
  while (iter != mem_free_blocks_.end()) {
    auto &cache_info = iter->second;
    if (cache_info->data_type == info.data_type &&
        cache_info->dims[0] >= info.dims[0] &&
        cache_info->dims[1] >= info.dims[1]) {
      break;
    }
    ++iter;
  }
  void *ptr = nullptr;
  if (iter == mem_free_blocks_.end()) {
    VLOG(3) << "OpenclImageManager::MemoryPool::ObtainMemory New memory: "
            << MakeString(info.dims);
    MACE_CHECK_SUCCESS(allocator_->New(info, &ptr));
    mem_used_blocks_.emplace(static_cast<index_t>(size),
                             std::make_shared<Buffer>(info, ptr));
  } else {
    VLOG(3) << "OpenclImageManager::MemoryPool::ObtainMemory Old memory: "
            << MakeString(info.dims);
    ptr = iter->second->mutable_memory<void>();
    mem_used_blocks_.emplace(iter->first, iter->second);
    mem_free_blocks_.erase(iter);
  }
  return ptr;
}

void OpenclImageManager::ImagePool::ReleaseMemory(void *ptr) {
  for (BlockList::iterator iter = mem_used_blocks_.begin();
       iter != mem_used_blocks_.end(); ++iter) {
    if (iter->second->memory<void>() == ptr) {
      mem_free_blocks_.emplace(iter->first, iter->second);
      mem_used_blocks_.erase(iter);
      return;
    }
  }
  LOG(FATAL) << "OpenclImageManager::ReleaseMemory,"
             << " but find an unknown ptr: " << ptr;
}

void OpenclImageManager::ImagePool::ReleaseAllMemory(bool del_buf) {
  for (BlockList::iterator iter = mem_used_blocks_.begin();
       iter != mem_used_blocks_.end(); ++iter) {
    mem_free_blocks_.emplace(iter->first, iter->second);
    VLOG(3) << "ReleaseAllMemory, ptr: " << iter->second;
  }
  mem_used_blocks_.clear();

  if (del_buf) {
    ClearMemory();
  }
}

}  // namespace mace


