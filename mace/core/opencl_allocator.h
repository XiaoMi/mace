//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_RUNTIME_OPENCL_OPENCL_ALLOCATOR_H_
#define MACE_CORE_RUNTIME_OPENCL_OPENCL_ALLOCATOR_H_

#include "mace/core/allocator.h"

namespace mace {

class OpenCLAllocator : public Allocator {
 public:
  OpenCLAllocator();

  ~OpenCLAllocator() override;

  void *New(size_t nbytes) override;

  void Delete(void *buffer) override;

  void *Map(void *buffer, size_t nbytes) override;

  void Unmap(void *buffer, void *mapped_ptr) override;

  bool OnHost() override;
};

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_ALLOCATOR_H_
