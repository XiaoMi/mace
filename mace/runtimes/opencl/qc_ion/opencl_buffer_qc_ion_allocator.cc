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

#include "mace/runtimes/opencl/qc_ion/opencl_buffer_qc_ion_allocator.h"

#include <memory>

#include "mace/core/runtime_failure_mock.h"
#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"

namespace mace {

OpenclBufferQcIonAllocator::OpenclBufferQcIonAllocator(
    OpenclExecutor *opencl_executor, std::shared_ptr<Rpcmem> rpcmem)
    : OpenclBaseQcIonAllocator(opencl_executor, rpcmem) {}

MemoryType OpenclBufferQcIonAllocator::GetMemType() {
  return MemoryType::GPU_BUFFER;
}

MaceStatus OpenclBufferQcIonAllocator::New(const MemInfo &info, void **result) {
  MACE_CHECK(info.mem_type == MemoryType::GPU_BUFFER);
  auto nbytes = info.bytes();
  if (nbytes == 0) {
    return MaceStatus::MACE_SUCCESS;
  }
  VLOG(1) << "Allocate OpenCL ION buffer: " << nbytes;

  if (ShouldMockRuntimeFailure()) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  cl_int error = CL_SUCCESS;
  cl_mem_ion_host_ptr ion_host;
  CreateQualcommBufferIONHostPtr(cpu_ion_allocator_.get(), nbytes, &ion_host);
  cl::Buffer *buffer = new cl::Buffer(
      opencl_executor_->context(),
      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
      nbytes, &ion_host, &error);

  if (error != CL_SUCCESS) {
    LOG(WARNING) << "Allocate OpenCL ION Buffer with "
                 << nbytes << " bytes failed because of "
                 << OpenCLErrorToString(error);
    delete buffer;
    *result = nullptr;
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  } else {
    cl_to_host_map_[static_cast<void *>(buffer)] = ion_host.ion_hostptr;
    *result = buffer;
    return MaceStatus::MACE_SUCCESS;
  }
}

void OpenclBufferQcIonAllocator::Delete(void *buffer) {
  MACE_LATENCY_LOGGER(1, "Free OpenCL ION buffer");
  MACE_CHECK(opencl_executor_->ion_type() == IONType::QUALCOMM_ION);
  if (buffer != nullptr) {
    auto it = cl_to_host_map_.find(buffer);
    MACE_CHECK(it != cl_to_host_map_.end(), "OpenCL ION buffer not found!");
    rpcmem_->Delete(it->second);
    cl_to_host_map_.erase(buffer);

    cl::Buffer *cl_buffer = static_cast<cl::Buffer *>(buffer);
    delete cl_buffer;
  }
}

}  // namespace mace
