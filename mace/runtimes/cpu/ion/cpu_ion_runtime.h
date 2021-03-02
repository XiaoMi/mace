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


#ifndef MACE_RUNTIMES_CPU_ION_CPU_ION_RUNTIME_H_
#define MACE_RUNTIMES_CPU_ION_CPU_ION_RUNTIME_H_

#include <memory>

#include "mace/core/memory/general_memory_manager.h"
#include "mace/runtimes/cpu/ion/cpu_ion_allocator.h"
#include "mace/runtimes/cpu/cpu_runtime.h"

namespace mace {
class CpuIonRuntime : public CpuRuntime {
 public:
  explicit CpuIonRuntime(RuntimeContext *runtime_context);
  ~CpuIonRuntime() = default;

 protected:
  MemoryManager *GetMemoryManager(MemoryType mem_type) override;

 private:
  std::shared_ptr<Rpcmem> rpcmem_;
  std::unique_ptr<CpuIonAllocator> buffer_ion_allocator_;
  std::unique_ptr<GeneralMemoryManager> buffer_ion_manager_;
};

}  // namespace mace

#endif  // MACE_RUNTIMES_CPU_ION_CPU_ION_RUNTIME_H_
