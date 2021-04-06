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

#include "mace/runtimes/apu/ion/apu_ion_runtime.h"

#include <memory>

#include "mace/core/runtime/runtime_context.h"
#include "mace/core/runtime/runtime_registry.h"
#include "mace/runtimes/cpu/ion/cpu_ion_allocator.h"

namespace mace {

ApuIonRuntime::ApuIonRuntime(RuntimeContext *runtime_context)
    :ApuRuntime(runtime_context) {
  MACE_CHECK(runtime_context->context_type == RuntimeContextType::RCT_ION);
  auto *ion_runtime_context = static_cast<IonRuntimeContext *>(runtime_context);
  rpcmem_ = ion_runtime_context->rpcmem;
}

ApuIonRuntime::~ApuIonRuntime() {}

RuntimeSubType ApuIonRuntime::GetRuntimeSubType() {
    return RuntimeSubType::RT_SUB_ION;
}

std::shared_ptr<Rpcmem> ApuIonRuntime::GetRpcmem() {
  return rpcmem_;
}

std::unique_ptr<Allocator> ApuIonRuntime::CreateAllocator() {
  return make_unique<CpuIonAllocator>(rpcmem_);
}

void RegisterApuIonRuntime(RuntimeRegistry *runtime_registry) {
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_APU,
                        RuntimeSubType::RT_SUB_ION, ApuIonRuntime);
}

}  // namespace mace
