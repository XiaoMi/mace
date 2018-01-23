//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_RUNTIME_OPENCL_PREALLOCATED_POOLED_ALLOCATOR_H_
#define MACE_CORE_RUNTIME_OPENCL_PREALLOCATED_POOLED_ALLOCATOR_H_

#include "mace/core/preallocated_pooled_allocator.h"
#include <unordered_map>

namespace mace {

class OpenCLPreallocatedPooledAllocator : public PreallocatedPooledAllocator {
 public:
  OpenCLPreallocatedPooledAllocator();

  ~OpenCLPreallocatedPooledAllocator() override;

  void PreallocateImage(int mem_id,
                        const std::vector<size_t> &image_shape,
                        DataType data_type) override;

  inline void *GetImage(int mem_id) override {
    MACE_CHECK(HasImage(mem_id), "image does not exist");
    return images_[mem_id].get();
  }

  inline bool HasImage(int mem_id) override {
    return images_.find(mem_id) != images_.end();
  }

  inline std::vector<size_t> GetImageSize(int mem_id) override {
    return image_shapes_[mem_id];
  }

 private:
  std::unordered_map<int, std::unique_ptr<void, std::function<void(void *)>>>
    images_;
  std::unordered_map<int, std::vector<size_t>> image_shapes_;
  Allocator *allocator;
};

} // namepsace mace

#endif // MACE_CORE_RUNTIME_OPENCL_PREALLOCATED_POOLED_ALLOCATOR_H_
