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

#include <stdlib.h>
#include <string.h>
#include <map>
#include <limits>
#include <vector>
#include <cstring>

#include "mace/core/macros.h"
#include "mace/core/types.h"
#include "mace/core/runtime_failure_mock.h"
#include "mace/public/mace.h"
#include "mace/utils/logging.h"

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

class Allocator {
 public:
  Allocator() {}
  virtual ~Allocator() noexcept {}
  virtual MaceStatus New(size_t nbytes, void **result) const = 0;
  virtual MaceStatus NewImage(const std::vector<size_t> &image_shape,
                              const DataType dt,
                              void **result) const = 0;
  virtual void Delete(void *data) const = 0;
  virtual void DeleteImage(void *data) const = 0;
  virtual void *Map(void *buffer, size_t offset, size_t nbytes) const = 0;
  virtual void *MapImage(void *buffer,
                         const std::vector<size_t> &image_shape,
                         std::vector<size_t> *mapped_image_pitch) const = 0;
  virtual void Unmap(void *buffer, void *mapper_ptr) const = 0;
  virtual bool OnHost() const = 0;
};

class CPUAllocator : public Allocator {
 public:
  ~CPUAllocator() override {}
  MaceStatus New(size_t nbytes, void **result) const override {
    VLOG(3) << "Allocate CPU buffer: " << nbytes;
    if (nbytes == 0) {
      return MaceStatus::MACE_SUCCESS;
    }

    if (ShouldMockRuntimeFailure()) {
      return MaceStatus::MACE_OUT_OF_RESOURCES;
    }

    void *data = nullptr;
#if defined(__ANDROID__) || defined(__hexagon__)
    data = memalign(kMaceAlignment, nbytes);
    if (data == NULL) {
      LOG(WARNING) << "Allocate CPU Buffer with "
                   << nbytes << " bytes failed because of"
                   << strerror(errno);
      *result = nullptr;
      return MaceStatus::MACE_OUT_OF_RESOURCES;
    }
#else
    int ret = posix_memalign(&data, kMaceAlignment, nbytes);
    if (ret != 0) {
      LOG(WARNING) << "Allocate CPU Buffer with "
                   << nbytes << " bytes failed because of"
                   << strerror(errno);
      if (data != NULL) {
        free(data);
      }
      *result = nullptr;
      return MaceStatus::MACE_OUT_OF_RESOURCES;
    }
#endif
    // TODO(heliangliang) This should be avoided sometimes
    memset(data, 0, nbytes);
    *result = data;
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus NewImage(const std::vector<size_t> &shape,
                      const DataType dt,
                      void **result) const override {
    MACE_UNUSED(shape);
    MACE_UNUSED(dt);
    MACE_UNUSED(result);
    LOG(FATAL) << "Allocate CPU image";
    return MaceStatus::MACE_SUCCESS;
  }

  void Delete(void *data) const override {
    MACE_CHECK_NOTNULL(data);
    VLOG(3) << "Free CPU buffer";
    free(data);
  }
  void DeleteImage(void *data) const override {
    LOG(FATAL) << "Free CPU image";
    free(data);
  };
  void *Map(void *buffer, size_t offset, size_t nbytes) const override {
    MACE_UNUSED(nbytes);
    return reinterpret_cast<char*>(buffer) + offset;
  }
  void *MapImage(void *buffer,
                 const std::vector<size_t> &image_shape,
                 std::vector<size_t> *mapped_image_pitch) const override {
    MACE_UNUSED(image_shape);
    MACE_UNUSED(mapped_image_pitch);
    return buffer;
  }
  void Unmap(void *buffer, void *mapper_ptr) const override {
    MACE_UNUSED(buffer);
    MACE_UNUSED(mapper_ptr);
  }
  bool OnHost() const override { return true; }
};

// Global CPU allocator used for CPU/GPU/DSP
Allocator *GetCPUAllocator();

}  // namespace mace

#endif  // MACE_CORE_ALLOCATOR_H_
