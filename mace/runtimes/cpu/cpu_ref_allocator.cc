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

#include "mace/runtimes/cpu/cpu_ref_allocator.h"

#include "mace/core/runtime_failure_mock.h"
#include "mace/utils/logging.h"

namespace mace {

MemoryType CpuRefAllocator::GetMemType() {
  return MemoryType::CPU_BUFFER;
}

MaceStatus CpuRefAllocator::New(const MemInfo &info, void **result) {
  auto nbytes = info.bytes();
  VLOG(3) << "Allocate CPU buffer: " << nbytes;
  if (nbytes <= 0) {
    return MaceStatus::MACE_SUCCESS;
  }

  if (ShouldMockRuntimeFailure()) {
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  MACE_RETURN_IF_ERROR(Memalign(result, kMaceAlignment, nbytes));

  return MaceStatus::MACE_SUCCESS;
}

void CpuRefAllocator::Delete(void *data) {
  MACE_CHECK_NOTNULL(data);
  VLOG(3) << "Free CPU buffer";
  free(data);
}

}  // namespace mace
