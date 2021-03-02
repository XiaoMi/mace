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

#ifndef MACE_CORE_MEMORY_SLICE_H_
#define MACE_CORE_MEMORY_SLICE_H_

#include <vector>

#include "mace/core/memory/buffer.h"

namespace mace {

class Slice : public Buffer {
 public:
  explicit Slice(
      const MemoryType buffer_mt, DataType dt,
      const std::vector<index_t> buffer_dims = std::vector<index_t>(),
      void *base_ptr = nullptr, index_t offset_bytes = 0)
      : Buffer(buffer_mt, dt, buffer_dims, base_ptr),
        buf_offset(offset_bytes) {}

  index_t offset() override {
    return buf_offset;
  }

 private:
  index_t buf_offset;
};

}  // namespace mace

#endif  // MACE_CORE_MEMORY_SLICE_H_


