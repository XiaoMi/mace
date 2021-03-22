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

#include "mace/runtimes/cpu/ion/cpu_ion_runtime.h"

#include <memory>

#include "mace/core/memory/buffer.h"
#include "mace/core/runtime/runtime_context.h"
#include "mace/core/runtime/runtime_registry.h"
#include "mace/utils/memory.h"

namespace mace {

CpuIonRuntime::CpuIonRuntime(RuntimeContext *runtime_context)
    : CpuRuntime(runtime_context) {
  MACE_CHECK(runtime_context->context_type == RuntimeContextType::RCT_ION);
  auto *ion_runtime_context =
      static_cast<IonRuntimeContext *>(runtime_context);
  rpcmem_ = ion_runtime_context->rpcmem;
  buffer_ion_allocator_ = make_unique<CpuIonAllocator>(rpcmem_),
  buffer_ion_manager_ =
      make_unique<GeneralMemoryManager>(buffer_ion_allocator_.get());
}

RuntimeSubType CpuIonRuntime::GetRuntimeSubType() {
    return RuntimeSubType::RT_SUB_ION;
}

MemoryManager *CpuIonRuntime::GetMemoryManager(MemoryType mem_type) {
  MemoryManager *buffer_manager = nullptr;
  if (mem_type == MemoryType::CPU_BUFFER) {
    buffer_manager = buffer_ion_manager_.get();
  } else {
    MACE_CHECK(false, "CpuIonRuntime::GetMemoryManagerByMemType",
               " find an invalid mem type:", mem_type);
  }

  return buffer_manager;
}

void RegisterCpuIonRuntime(RuntimeRegistry *runtime_registry) {
  MACE_UNUSED(runtime_registry);
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_CPU,
                        RuntimeSubType::RT_SUB_ION, CpuIonRuntime);
}

}  // namespace mace
