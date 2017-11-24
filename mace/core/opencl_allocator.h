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

  /*
   * Only support shape.size() > 1 and collapse first n-2 dimensions to depth.
   * Use Image3D with RGBA (128-bit) format to represent the image.
   *
   * @ shape : [depth, ..., height, width ].
   */
  void *NewImage(const std::vector<size_t> &image_shape,
                 const DataType dt) override;

  void Delete(void *buffer) override;

  void DeleteImage(void *buffer) override;

  void *Map(void *buffer, size_t nbytes) override;

  void *MapImage(void *buffer,
                 const std::vector<size_t> &image_shape,
                 std::vector<size_t> &mapped_image_pitch) override;

  void Unmap(void *buffer, void *mapped_ptr) override;

  bool OnHost() override;
};

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_ALLOCATOR_H_
