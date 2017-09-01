//
// Created by liyin on 8/28/17.
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_ALLOCATOR_H_
#define MACE_CORE_ALLOCATOR_H_

#include <malloc.h>
#include "mace/core/common.h"
#include "mace/proto/mace.pb.h"

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
  virtual void* New(size_t nbytes) = 0;
  virtual void Delete(void* data) = 0;
  virtual void CopyBytes(void* dst, const void* src, size_t size) = 0;

  template <typename T>
  T* New(size_t num_elements) {
    if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T))) {
      return NULL;
    }
    void* p = New(sizeof(T) * num_elements);
    T* typed_p = reinterpret_cast<T*>(p);
    return typed_p;
  }
};

class CPUAllocator: public Allocator {
 public:
  ~CPUAllocator() override {}
  void* New(size_t nbytes) override {
    void* data = nullptr;
#ifdef __ANDROID__
    data = memalign(kMaceAlignment, nbytes);
#else
    CHECK(posix_memalign(&data, kMaceAlignment, nbytes) == 0);
#endif
    CHECK_NOTNULL(data);
    // TODO(heliangliang) This should be avoided sometimes
    memset(data, 0, nbytes);
    return data;
  }

  void Delete(void* data) {
    free(data);
  }

  void CopyBytes(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
  }
};

// Get the CPU Alloctor.
CPUAllocator* cpu_allocator();
// Sets the CPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
void SetCPUAllocator(CPUAllocator* alloc);

template <DeviceType D>
struct DeviceContext {};

template <>
struct DeviceContext<DeviceType::CPU> {
  static Allocator* allocator() { return cpu_allocator(); }
};

template <>
struct DeviceContext<DeviceType::NEON> {
  static Allocator* allocator() { return cpu_allocator(); }
};

Allocator* GetDeviceAllocator(DeviceType type);

} // namespace mace

#endif // MACE_CORE_ALLOCATOR_H_
