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

#ifndef MACE_RUNTIMES_OPENCL_QC_ION_OPENCL_BASE_QC_ION_ALLOCATOR_H_
#define MACE_RUNTIMES_OPENCL_QC_ION_OPENCL_BASE_QC_ION_ALLOCATOR_H_

#include <memory>
#include <unordered_map>

#include "mace/core/memory/allocator.h"
#include "mace/runtimes/cpu/ion/cpu_ion_allocator.h"
#include "mace/runtimes/opencl/core/cl2_header.h"

namespace mace {

class OpenclExecutor;

class OpenclBaseQcIonAllocator : public Allocator {
 public:
  explicit OpenclBaseQcIonAllocator(OpenclExecutor *opencl_executor,
                                    std::shared_ptr<Rpcmem> rpcmem);
  virtual ~OpenclBaseQcIonAllocator() {}

  void *GetMappedPtrByIonBuffer(void *buffer);
  std::shared_ptr<Rpcmem> GetRpcmem();

 protected:
  void CreateQualcommBufferIONHostPtr(
      CpuIonAllocator *cpu_ion_allocator,
      const index_t nbytes, cl_mem_ion_host_ptr *ion_host);

 protected:
  std::unique_ptr<CpuIonAllocator> cpu_ion_allocator_;
  std::unordered_map<void *, void *> cl_to_host_map_;
  OpenclExecutor *opencl_executor_;
  std::shared_ptr<Rpcmem> rpcmem_;
};

}  // namespace mace

#endif  // MACE_RUNTIMES_OPENCL_QC_ION_OPENCL_BASE_QC_ION_ALLOCATOR_H_
