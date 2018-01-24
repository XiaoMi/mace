//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_PREALLOCATED_POOLED_ALLOCATOR_H_
#define MACE_CORE_PREALLOCATED_POOLED_ALLOCATOR_H_

#include "mace/core/allocator.h"

namespace mace {

class PreallocatedPooledAllocator {
 public:
  PreallocatedPooledAllocator() {}

  virtual ~PreallocatedPooledAllocator() noexcept {}

  virtual void PreallocateImage(int mem_id,
                                const std::vector<size_t> &image_shape,
                                DataType data_type) = 0;

  virtual void *GetImage(int mem_id) = 0;

  virtual bool HasImage(int mem_id) = 0;

  virtual std::vector<size_t> GetImageSize(int mem_id) = 0;
};

} // namespace mace

#endif // MACE_CORE_PREALLOCATED_POOLED_ALLOCATOR_H_
