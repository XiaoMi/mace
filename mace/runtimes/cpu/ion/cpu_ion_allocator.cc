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

#include "mace/runtimes/cpu/ion/cpu_ion_allocator.h"

#include <memory>

#include "mace/port/port.h"
#include "third_party/rpcmem/rpcmem.h"

namespace mace {

CpuIonAllocator::CpuIonAllocator(std::shared_ptr<Rpcmem> rpcmem)
    : rpcmem_(rpcmem) {}

MemoryType CpuIonAllocator::GetMemType() {
  return MemoryType::CPU_BUFFER;
}

MaceStatus CpuIonAllocator::New(const MemInfo &info, void **result) {
  int nbytes = info.bytes();
  *result = New(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_CACHED, nbytes);
  MACE_CHECK(*result != nullptr, "info.dims: ", MakeString(info.dims));

  return MaceStatus::MACE_SUCCESS;
}

void *CpuIonAllocator::New(int heapid, uint32_t flags, int nbytes) {
  return rpcmem_->New(heapid, flags, PadAlignSize(nbytes));
}

void CpuIonAllocator::Delete(void *data) {
  rpcmem_->Delete(data);
}

std::shared_ptr<Rpcmem> CpuIonAllocator::GetRpcmem() {
  return rpcmem_;
}

}  // namespace mace
