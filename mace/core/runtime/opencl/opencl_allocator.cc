//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_allocator.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {

OpenCLAllocator::OpenCLAllocator() {}

OpenCLAllocator::~OpenCLAllocator() {}
void *OpenCLAllocator::New(size_t nbytes) {
  cl_int error;
  cl::Buffer *buffer = new cl::Buffer(OpenCLRuntime::Get()->context(),
                                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                      nbytes, nullptr, &error);
  MACE_CHECK(error == CL_SUCCESS);
  MACE_CHECK_NOTNULL(buffer);
  return static_cast<void *>(buffer);
}

void OpenCLAllocator::Delete(void *buffer) {
  if (buffer != nullptr) {
    cl::Buffer *cl_buffer = static_cast<cl::Buffer *>(buffer);
    delete cl_buffer;
  }
}

void *OpenCLAllocator::Map(void *buffer, size_t nbytes) {
  auto cl_buffer = static_cast<cl::Buffer *>(buffer);
  auto queue = OpenCLRuntime::Get()->command_queue();
  // TODO(heliangliang) Non-blocking call
  cl_int error;
  void *mapped_ptr =
      queue.enqueueMapBuffer(*cl_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
                             nbytes, nullptr, nullptr, &error);
  MACE_CHECK(error == CL_SUCCESS);
  return mapped_ptr;
}

void OpenCLAllocator::Unmap(void *buffer, void *mapped_ptr) {
  auto cl_buffer = static_cast<cl::Buffer *>(buffer);
  auto queue = OpenCLRuntime::Get()->command_queue();
  MACE_CHECK(queue.enqueueUnmapMemObject(*cl_buffer, mapped_ptr, nullptr,
                                         nullptr) == CL_SUCCESS);
}

bool OpenCLAllocator::OnHost() { return false; }

MACE_REGISTER_ALLOCATOR(DeviceType::OPENCL, new OpenCLAllocator());

}  // namespace mace
