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

#ifndef MACE_CORE_MEMORY_ALLOCATOR_H_
#define MACE_CORE_MEMORY_ALLOCATOR_H_

#include <functional>
#include <numeric>
#include <vector>

#include "mace/public/mace.h"
#include "mace/core/types.h"
#include "mace/proto/mace.pb.h"

namespace mace {

#if defined(__hexagon__)
constexpr size_t kMaceAlignment = 128;
#elif defined(__ANDROID__)
// arm cache line
constexpr size_t kMaceAlignment = 64;
#else
// 32 bytes = 256 bits (AVX512)
constexpr size_t kMaceAlignment = 32;
#endif

inline index_t PadAlignSize(index_t size) {
  return (size + kMaceAlignment - 1) & (~(kMaceAlignment - 1));
}

/**
 * RENT_PRIVATE: used for const tensor
 * RENT_SHARE: shared by ops' output tensor
 * RENT_SCRATCH: used for temp buffer or tensor
 * RENT_SLICE: used for slice buffer
 */
enum BufRentType {
  RENT_PRIVATE,
  RENT_SHARE,
  RENT_SCRATCH,
  RENT_SLICE,
};

struct MemInfo {
  MemoryType mem_type;
  DataType data_type;
  std::vector<index_t> dims;

  explicit MemInfo(const MemoryType memory_type, const DataType dt,
                   const std::vector<index_t> &mem_dims)
      : mem_type(memory_type), data_type(dt), dims(mem_dims) {}

  template <typename T>
  static std::vector<index_t> IndexT(const std::vector<T> &in_dims) {
    std::vector<index_t> index_dims;
    for (auto i = in_dims.begin(); i != in_dims.end(); ++i) {
      index_dims.push_back(static_cast<index_t>(*i));
    }
    return index_dims;
  }

  index_t size() const {
    return std::accumulate(dims.begin(), dims.end(),
                           1, std::multiplies<index_t>());
  }

  index_t bytes() const {
    return size() * GetEnumTypeSize(data_type);
  }

  virtual ~MemInfo() {}
};

class Allocator {
 public:
  Allocator() {}
  virtual ~Allocator() noexcept {}

  virtual MemoryType GetMemType() = 0;

  virtual MaceStatus New(const MemInfo &info, void **result) = 0;

  virtual void Delete(void *data) = 0;
};

}  // namespace mace

#endif  // MACE_CORE_MEMORY_ALLOCATOR_H_
