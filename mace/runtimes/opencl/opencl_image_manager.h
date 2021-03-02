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

#ifndef MACE_RUNTIMES_OPENCL_OPENCL_IMAGE_MANAGER_H_
#define MACE_RUNTIMES_OPENCL_OPENCL_IMAGE_MANAGER_H_

#include <string>

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "mace/core/memory/buffer.h"
#include "mace/core/memory/memory_manager.h"
#include "mace/runtimes/opencl/opencl_image_allocator.h"

namespace mace {

class OpenclImageManager : public MemoryManager {
 public:
  explicit OpenclImageManager(Allocator *allocator);
  ~OpenclImageManager();

  void *ObtainMemory(const MemInfo &info, const BufRentType rent_type) override;
  void ReleaseMemory(void *ptr, const BufRentType rent_type) override;
  std::vector<index_t> GetMemoryRealSize(const void *ptr) override;
  void ReleaseAllMemory(const BufRentType rent_type, bool del_buf) override;

 private:
  typedef std::multimap<index_t, std::shared_ptr<Buffer>> BlockList;
  class ImagePool {
   public:
    explicit ImagePool(Allocator *allocator);
    ~ImagePool();

    void *ObtainMemory(const MemInfo &info);
    void ReleaseMemory(void *ptr);
    std::vector<index_t> GetMemoryRealSize(const void *ptr);
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
  typedef std::unordered_map<int, std::unique_ptr<ImagePool>> SharedPools;
  SharedPools shared_pools_;
};

}  // namespace mace



#endif  // MACE_RUNTIMES_OPENCL_OPENCL_IMAGE_MANAGER_H_
