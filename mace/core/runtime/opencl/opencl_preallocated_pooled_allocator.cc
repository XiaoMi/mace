//
// Copyright (c) 2018 XiaoMi All rights reserved.
//

#include "mace/core/runtime/opencl/opencl_preallocated_pooled_allocator.h"

namespace mace {

OpenCLPreallocatedPooledAllocator::OpenCLPreallocatedPooledAllocator()
  : allocator(GetDeviceAllocator(DeviceType::OPENCL)) {
}

OpenCLPreallocatedPooledAllocator::~OpenCLPreallocatedPooledAllocator() {
}

void OpenCLPreallocatedPooledAllocator::PreallocateImage(int mem_id,
                                                         const std::vector<
                                                           size_t> &image_shape,
                                                         DataType data_type) {
  MACE_CHECK(!this->HasImage(mem_id), "Memory already exists: ", mem_id);
  VLOG(2) << "Preallocate OpenCL image: " << mem_id << " "
          << image_shape[0] << ", " << image_shape[1];
  images_[mem_id] = std::move(std::unique_ptr<void, std::function<void(void *)>>(
    allocator->NewImage(image_shape, data_type), [this](void *p) {
      this->allocator->DeleteImage(p);
    }));
  image_shapes_[mem_id] = image_shape;
}

} // namespace mace
