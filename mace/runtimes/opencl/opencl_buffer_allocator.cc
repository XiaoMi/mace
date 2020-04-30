// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/runtimes/opencl/opencl_buffer_allocator.h"

#include "mace/core/runtime_failure_mock.h"
#include "mace/runtimes/opencl/core/opencl_executor.h"

namespace mace {

OpenclBufferAllocator::OpenclBufferAllocator(OpenclExecutor *opencl_executor)
    : opencl_executor_(opencl_executor) {}

MemoryType OpenclBufferAllocator::GetMemType() {
  return MemoryType::GPU_BUFFER;
}

MaceStatus OpenclBufferAllocator::New(const MemInfo &info, void **result) {
  MACE_CHECK(info.mem_type == MemoryType::GPU_BUFFER);
  auto nbytes = info.bytes();
  if (nbytes == 0) {
    return MaceStatus::MACE_SUCCESS;
  }
  VLOG(3) << "Allocate OpenCL buffer: " << nbytes;

  if (ShouldMockRuntimeFailure()) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  cl_int error = CL_SUCCESS;
  cl::Buffer *buffer = new cl::Buffer(opencl_executor_->context(),
                                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                      nbytes, nullptr, &error);

  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL Buffer with "
                 << nbytes << " bytes failed because of "
                 << OpenCLErrorToString(error);
    delete buffer;
    *result = nullptr;
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  } else {
    *result = buffer;
    return MaceStatus::MACE_SUCCESS;
  }
}

void OpenclBufferAllocator::Delete(void *buffer) {
  MACE_LATENCY_LOGGER(1, "Free OpenCL buffer");
  if (buffer != nullptr) {
    cl::Buffer *cl_buffer = static_cast<cl::Buffer *>(buffer);
    delete cl_buffer;
  }
}

}  // namespace mace
