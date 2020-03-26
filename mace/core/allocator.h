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

#ifndef MACE_CORE_ALLOCATOR_H_
#define MACE_CORE_ALLOCATOR_H_

#include <cstdlib>
#include <map>
#include <limits>
#include <vector>
#include <cstring>

#include "mace/utils/macros.h"
#include "mace/core/types.h"
#include "mace/core/runtime_failure_mock.h"
#include "mace/public/mace.h"
#include "mace/utils/logging.h"

#ifdef MACE_ENABLE_RPCMEM
#include "mace/core/rpcmem.h"
#endif  // MACE_ENABLE_RPCMEM

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

class Allocator {
 public:
  Allocator() {}
  virtual ~Allocator() noexcept {}
  virtual MaceStatus New(size_t nbytes, void **result) = 0;
  virtual MaceStatus NewImage(const std::vector<size_t> &image_shape,
                              const DataType dt,
                              void **result);
  virtual void Delete(void *data) = 0;
  virtual void DeleteImage(void *data);
  virtual void *Map(void *buffer,
                    size_t offset,
                    size_t nbytes,
                    bool finish_cmd_queue);
  virtual void *MapImage(void *buffer,
                         const std::vector<size_t> &image_shape,
                         std::vector<size_t> *mapped_image_pitch,
                         bool finish_cmd_queue);
  virtual void Unmap(void *buffer, void *mapper_ptr);
  virtual bool OnHost() const = 0;
#ifdef MACE_ENABLE_RPCMEM
  virtual Rpcmem *rpcmem();
#endif  // MACE_ENABLE_RPCMEM
};

class CPUAllocator : public Allocator {
 public:
  ~CPUAllocator() override {}
  MaceStatus New(size_t nbytes, void **result) override {
    VLOG(3) << "Allocate CPU buffer: " << nbytes;
    if (nbytes == 0) {
      return MaceStatus::MACE_SUCCESS;
    }

    if (ShouldMockRuntimeFailure()) {
      return MaceStatus::MACE_OUT_OF_RESOURCES;
    }

    MACE_RETURN_IF_ERROR(Memalign(result, kMaceAlignment, nbytes));
    // TODO(heliangliang) This should be avoided sometimes
    memset(*result, 0, nbytes);
    return MaceStatus::MACE_SUCCESS;
  }

  void Delete(void *data) override {
    MACE_CHECK_NOTNULL(data);
    VLOG(3) << "Free CPU buffer";
    free(data);
  }

  bool OnHost() const override { return true; }
};

// Global CPU allocator used for CPU/GPU/DSP
Allocator *GetCPUAllocator();

}  // namespace mace

#endif  // MACE_CORE_ALLOCATOR_H_
