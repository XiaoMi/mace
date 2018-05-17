//
// Created by liyin on 8/28/17.
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_ALLOCATOR_H_
#define MACE_CORE_ALLOCATOR_H_

#include <malloc.h>
#include <map>
#include <limits>
#include <vector>

#include "mace/core/registry.h"
#include "mace/core/types.h"
#include "mace/public/mace.h"

namespace mace {

#ifdef __ANDROID__
// 16 bytes = 128 bits = 32 * 4 (Neon)
constexpr size_t kMaceAlignment = 16;
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
    void *data = nullptr;
#ifdef __ANDROID__
    data = memalign(kMaceAlignment, nbytes);
#else
    MACE_CHECK(posix_memalign(&data, kMaceAlignment, nbytes) == 0);
#endif
    MACE_CHECK_NOTNULL(data);
    // TODO(heliangliang) This should be avoided sometimes
    memset(data, 0, nbytes);
    *result = data;
    return MaceStatus::MACE_SUCCESS;
  }

  MaceStatus NewImage(const std::vector<size_t> &shape,
                      const DataType dt,
                      void **status) const override {
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
    return reinterpret_cast<char*>(buffer) + offset;
  }
  void *MapImage(void *buffer,
                 const std::vector<size_t> &image_shape,
                 std::vector<size_t> *mapped_image_pitch) const override {
    return buffer;
  }
  void Unmap(void *buffer, void *mapper_ptr) const override {}
  bool OnHost() const override { return true; }
};

std::map<int32_t, Allocator *> *gAllocatorRegistry();

Allocator *GetDeviceAllocator(DeviceType type);

struct AllocatorRegisterer {
  explicit AllocatorRegisterer(DeviceType type, Allocator *alloc) {
    if (gAllocatorRegistry()->count(type)) {
      LOG(ERROR) << "Allocator for device type " << type
                 << " registered twice. This should not happen."
                 << gAllocatorRegistry()->count(type);
      std::exit(1);
    }
    gAllocatorRegistry()->emplace(type, alloc);
  }
};

#define MACE_REGISTER_ALLOCATOR(type, alloc)                                  \
  namespace {                                                                 \
  static AllocatorRegisterer MACE_ANONYMOUS_VARIABLE(Allocator)(type, alloc); \
  }

}  // namespace mace

#endif  // MACE_CORE_ALLOCATOR_H_
