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

#ifndef MACE_CORE_PREALLOCATED_POOLED_ALLOCATOR_H_
#define MACE_CORE_PREALLOCATED_POOLED_ALLOCATOR_H_

#include <memory>
#include <utility>
#include <unordered_map>

#include "mace/core/allocator.h"
#include "mace/core/buffer.h"

namespace mace {

class PreallocatedPooledAllocator {
 public:
  PreallocatedPooledAllocator() {}

  ~PreallocatedPooledAllocator() noexcept {}

  void SetBuffer(int mem_id, std::unique_ptr<BufferBase> &&buffer) {
    buffers_[mem_id] = std::move(buffer);
  }

  BufferBase *GetBuffer(int mem_id) {
    if (buffers_.find(mem_id) != buffers_.end()) {
      return buffers_[mem_id].get();
    } else {
      return nullptr;
    }
  }

  virtual bool HasBuffer(int mem_id) {
    return buffers_.find(mem_id) != buffers_.end();
  }

 private:
  std::unordered_map<int, std::unique_ptr<BufferBase>> buffers_;
};

}  // namespace mace

#endif  // MACE_CORE_PREALLOCATED_POOLED_ALLOCATOR_H_
