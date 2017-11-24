//
// Created by liyin on 8/28/17.
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_ALLOCATOR_H_
#define MACE_CORE_ALLOCATOR_H_

#include <malloc.h>
#include "mace/core/common.h"
#include "mace/core/registry.h"
#include "mace/proto/mace.pb.h"
#include "mace/core/types.h"

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
  virtual void *New(size_t nbytes) = 0;
  virtual void *NewImage(const std::vector<size_t> &image_shape,
                         const DataType dt) = 0;
  virtual void Delete(void *data) = 0;
  virtual void DeleteImage(void *data) = 0;
  virtual void *Map(void *buffer, size_t nbytes) = 0;
  virtual void *MapImage(void *buffer,
                         const std::vector<size_t> &image_shape,
                         std::vector<size_t> &mapped_image_pitch) = 0;
  virtual void Unmap(void *buffer, void *mapper_ptr) = 0;
  virtual bool OnHost() = 0;

  template <typename T>
  T *New(size_t num_elements) {
    if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T))) {
      return nullptr;
    }
    void *p = New(sizeof(T) * num_elements);
    T *typed_p = reinterpret_cast<T *>(p);
    return typed_p;
  }
};

class CPUAllocator : public Allocator {
 public:
  ~CPUAllocator() override {}
  void *New(size_t nbytes) override {
    void *data = nullptr;
#ifdef __ANDROID__
    data = memalign(kMaceAlignment, nbytes);
#else
    MACE_CHECK(posix_memalign(&data, kMaceAlignment, nbytes) == 0);
#endif
    MACE_CHECK_NOTNULL(data);
    // TODO(heliangliang) This should be avoided sometimes
    memset(data, 0, nbytes);
    return data;
  }

  void *NewImage(const std::vector<size_t> &shape,
                 const DataType dt) override {
    return nullptr;
  }

  void Delete(void *data) override { free(data); }
  void DeleteImage(void *data) override { free(data); };
  void *Map(void *buffer, size_t nbytes) override { return buffer; }
  void *MapImage(void *buffer,
                 const std::vector<size_t> &image_shape,
                 std::vector<size_t> &mapped_image_pitch) override {
    return buffer;
  }
  void Unmap(void *buffer, void *mapper_ptr) override {}
  bool OnHost() override { return true; }
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
