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

namespace mace {

CpuIonAllocator::CpuIonAllocator(std::shared_ptr<Rpcmem> rpcmem)
    : rpcmem_(rpcmem) {}

MemoryType CpuIonAllocator::GetMemType() {
  return MemoryType::CPU_BUFFER;
}

MaceStatus CpuIonAllocator::New(const MemInfo &info, void **result) {
  int nbytes = info.bytes();
  *result = New(nbytes);
  MACE_CHECK(*result != nullptr, "info.dims: ", MakeString(info.dims));

  return MaceStatus::MACE_SUCCESS;
}

void *CpuIonAllocator::New(int nbytes) {
  // It seems that ion memory needs to add an extra segment of memory as
  // padding. I only know that can avoid occasional crash.
  nbytes += kMaceAlignment - 1;
  return rpcmem_->New(PadAlignSize(nbytes));
}

void CpuIonAllocator::Delete(void *data) {
  rpcmem_->Delete(data);
}

std::shared_ptr<Rpcmem> CpuIonAllocator::GetRpcmem() {
  return rpcmem_;
}

}  // namespace mace
