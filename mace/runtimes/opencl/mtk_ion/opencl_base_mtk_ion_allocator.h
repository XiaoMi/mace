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

#ifndef MACE_RUNTIMES_OPENCL_MTK_ION_OPENCL_BASE_MTK_ION_ALLOCATOR_H_
#define MACE_RUNTIMES_OPENCL_MTK_ION_OPENCL_BASE_MTK_ION_ALLOCATOR_H_

#include <memory>
#include <unordered_map>

#include "mace/core/memory/allocator.h"
#include "mace/runtimes/cpu/ion/cpu_ion_allocator.h"
#include "mace/runtimes/opencl/core/cl2_header.h"

namespace mace {

class OpenclExecutor;

class OpenclBaseMtkIonAllocator : public Allocator {
 public:
  explicit OpenclBaseMtkIonAllocator(OpenclExecutor *opencl_executor,
                                    std::shared_ptr<Rpcmem> rpcmem);
  virtual ~OpenclBaseMtkIonAllocator() {}

  void *GetMappedPtrByIonBuffer(void *buffer);
  std::shared_ptr<Rpcmem> GetRpcmem();

 protected:
  void CreateMtkIONPtr(
      CpuIonAllocator *cpu_ion_allocator,
      const index_t nbytes, void **host, cl_mem *ion_mem, cl_int *error);

 protected:
  std::unique_ptr<CpuIonAllocator> cpu_ion_allocator_;
  std::unordered_map<void *, void *> cl_to_host_map_;
  OpenclExecutor *opencl_executor_;
  std::shared_ptr<Rpcmem> rpcmem_;
  using clImportMemoryARMFunc = cl_mem (*)(cl_context context,
                                cl_mem_flags flags,
                                const cl_import_properties_arm *properties,
                                void *memory,
                                size_t size,
                                cl_int *errorcode_ret);
  clImportMemoryARMFunc clImportMemoryARM = nullptr;
};

}  // namespace mace

#endif  // MACE_RUNTIMES_OPENCL_MTK_ION_OPENCL_BASE_MTK_ION_ALLOCATOR_H_
