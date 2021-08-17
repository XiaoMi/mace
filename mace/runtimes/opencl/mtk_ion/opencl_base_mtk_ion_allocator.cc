// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include "mace/runtimes/opencl/mtk_ion/opencl_base_mtk_ion_allocator.h"

#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/mtk_ion/opencl_mtk_ion_executor.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"

namespace mace {

OpenclBaseMtkIonAllocator::OpenclBaseMtkIonAllocator(
    OpenclExecutor *opencl_executor, std::shared_ptr<Rpcmem> rpcmem)
    : cpu_ion_allocator_(make_unique<CpuIonAllocator>(rpcmem)),
      opencl_executor_(opencl_executor), rpcmem_(rpcmem) {
  clImportMemoryARM = (clImportMemoryARMFunc) clGetExtensionFunctionAddressForPlatform(opencl_executor->platform()(),
                                                                                       "clImportMemoryARM");
  MACE_CHECK(clImportMemoryARM != nullptr, "can't get extension functionclImportMemoryARM");
}

std::shared_ptr<Rpcmem> OpenclBaseMtkIonAllocator::GetRpcmem() {
  return rpcmem_;
}

void OpenclBaseMtkIonAllocator::CreateMtkIONPtr(
    CpuIonAllocator *cpu_ion_allocator,
    const index_t nbytes, void **host, cl_mem *ion_mem, cl_int *error) {
  MemInfo mem_info(MemoryType::CPU_BUFFER, DataType::DT_UINT8, {nbytes});
  MACE_CHECK_SUCCESS(cpu_ion_allocator->New(mem_info, host));
  int fd = rpcmem_->ToFd(*host);
  MACE_CHECK(fd >= 0, "Invalid rpcmem file descriptor: ", fd, ", ", *host);
  static const cl_import_properties_arm prop[3] = {CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0};
  *ion_mem = clImportMemoryARM(opencl_executor_->context().get(), CL_MEM_READ_WRITE, prop, &fd, nbytes, error);
}

void *OpenclBaseMtkIonAllocator::GetMappedPtrByIonBuffer(void *buffer) {
  auto it = cl_to_host_map_.find(buffer);
  if (it != cl_to_host_map_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

}  // namespace mace
