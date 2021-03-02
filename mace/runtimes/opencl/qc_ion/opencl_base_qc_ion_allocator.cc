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

#include "mace/runtimes/opencl/qc_ion/opencl_base_qc_ion_allocator.h"

#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/qc_ion/opencl_qc_ion_executor.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"

namespace mace {

OpenclBaseQcIonAllocator::OpenclBaseQcIonAllocator(
    OpenclExecutor *opencl_executor, std::shared_ptr<Rpcmem> rpcmem)
    : cpu_ion_allocator_(make_unique<CpuIonAllocator>(rpcmem)),
      opencl_executor_(opencl_executor), rpcmem_(rpcmem) {}

std::shared_ptr<Rpcmem> OpenclBaseQcIonAllocator::GetRpcmem() {
  return rpcmem_;
}

void OpenclBaseQcIonAllocator::CreateQualcommBufferIONHostPtr(
    CpuIonAllocator *cpu_ion_allocator,
    const index_t nbytes, cl_mem_ion_host_ptr *ion_host) {
  auto *opencl_executor = OpenclQcIonExecutor::Get(opencl_executor_);
  auto nbytes_pad = nbytes + opencl_executor->qcom_ext_mem_padding();
  MemInfo mem_info(MemoryType::CPU_BUFFER, DataType::DT_UINT8, {nbytes_pad});
  void *host = nullptr;
  MACE_CHECK_SUCCESS(cpu_ion_allocator->New(mem_info, &host));
  auto host_addr = reinterpret_cast<std::uintptr_t>(host);
  auto page_size = opencl_executor->qcom_page_size();
  MACE_CHECK(host_addr % page_size == 0, "ION memory address: ", host_addr,
             " must be aligned to page size: ", page_size);
  int fd = rpcmem_->ToFd(host);
  MACE_CHECK(fd >= 0, "Invalid rpcmem file descriptor: ", fd, ", ", host);

  ion_host->ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
  ion_host->ext_host_ptr.host_cache_policy =
      opencl_executor->qcom_host_cache_policy();
  ion_host->ion_filedesc = fd;
  ion_host->ion_hostptr = host;
}

void *OpenclBaseQcIonAllocator::GetMappedPtrByIonBuffer(void *buffer) {
  auto it = cl_to_host_map_.find(buffer);
  if (it != cl_to_host_map_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

}  // namespace mace
