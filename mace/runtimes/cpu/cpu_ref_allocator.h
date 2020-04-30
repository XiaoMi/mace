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

#ifndef MACE_RUNTIMES_CPU_CPU_REF_ALLOCATOR_H_
#define MACE_RUNTIMES_CPU_CPU_REF_ALLOCATOR_H_

#include "mace/core/memory/allocator.h"

namespace mace {
class CpuRefAllocator : public Allocator {
 public:
  CpuRefAllocator() {}
  ~CpuRefAllocator() {}

  MemoryType GetMemType() override;
  MaceStatus New(const MemInfo &info, void **result) override;
  void Delete(void *data) override;
};

}  // namespace mace

#endif  // MACE_RUNTIMES_CPU_CPU_REF_ALLOCATOR_H_
