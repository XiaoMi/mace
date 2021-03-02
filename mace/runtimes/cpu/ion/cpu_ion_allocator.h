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


#ifndef MACE_RUNTIMES_CPU_ION_CPU_ION_ALLOCATOR_H_
#define MACE_RUNTIMES_CPU_ION_CPU_ION_ALLOCATOR_H_

#include "mace/core/memory/allocator.h"
#include "mace/core/memory/rpcmem/rpcmem.h"



namespace mace {
class CpuIonAllocator : public Allocator {
 public:
  CpuIonAllocator(std::shared_ptr<Rpcmem> rpcmem);
  ~CpuIonAllocator() = default;

  MemoryType GetMemType() override;
  MaceStatus New(const MemInfo &info, void **result) override;
  void Delete(void *data) override;

  // for ion
  void *New(int heapid, uint32_t flags, int nbytes);
  std::shared_ptr<Rpcmem> GetRpcmem();

 private:
  std::shared_ptr<Rpcmem> rpcmem_;
};
}  // namespace mace

#endif  // MACE_RUNTIMES_CPU_ION_CPU_ION_ALLOCATOR_H_
