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

#include "mace/core/runtime/runtime_registry.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/runtimes/opencl/core/opencl_executor.h"
#endif  // MACE_ENABLE_OPENCL

#ifdef MACE_ENABLE_RPCMEM
#include "mace/core/memory/rpcmem/rpcmem.h"
#endif  // MACE_ENABLE_RPCMEM

namespace mace {

extern void RegisterCpuRefRuntime(RuntimeRegistry *runtime_registry);

#ifdef MACE_ENABLE_RPCMEM
extern void RegisterCpuIonRuntime(RuntimeRegistry *runtime_registry);
#endif  // MACE_ENABLE_RPCMEM

#ifdef MACE_ENABLE_OPENCL
extern void RegisterOpenclRefRuntime(RuntimeRegistry *runtime_registry);
#ifdef MACE_ENABLE_RPCMEM
extern void RegisterOpenclQcIonRuntime(RuntimeRegistry *runtime_registry);
#endif  // MACE_ENABLE_RPCMEM
#endif  // MACE_ENABLE_OPENCL

#ifdef MACE_ENABLE_HEXAGON
extern void RegisterHexagonDspRuntime(RuntimeRegistry *runtime_registry);
#endif  // MACE_ENABLE_HEXAGON

#ifdef MACE_ENABLE_HTA
extern void RegisterHexagonHtaRuntime(RuntimeRegistry *runtime_registry);
#ifdef MACE_ENABLE_OPENCL
extern void RegisterHexagonHtaOpenclRuntime(RuntimeRegistry *runtime_registry);
#endif  // MACE_ENABLE_OPENCL
#endif  // MACE_ENABLE_HTA

#ifdef MACE_ENABLE_APU
extern void RegisterApuRuntime(RuntimeRegistry *runtime_registry);
#endif  // MACE_ENABLE_APU

void RegisterAllRuntimes(RuntimeRegistry *runtime_registry) {
  RegisterCpuRefRuntime(runtime_registry);

#ifdef MACE_ENABLE_RPCMEM
  RegisterCpuIonRuntime(runtime_registry);
#endif  // MACE_ENABLE_RPCMEM

#ifdef MACE_ENABLE_OPENCL
  RegisterOpenclRefRuntime(runtime_registry);
#ifdef MACE_ENABLE_RPCMEM
  RegisterOpenclQcIonRuntime(runtime_registry);
#endif  // MACE_ENABLE_RPCMEM
#endif  // MACE_ENABLE_OPENCL

#ifdef MACE_ENABLE_HEXAGON
  RegisterHexagonDspRuntime(runtime_registry);
#endif  // MACE_ENABLE_HEXAGON

#ifdef MACE_ENABLE_HTA
  RegisterHexagonHtaRuntime(runtime_registry);
#ifdef MACE_ENABLE_OPENCL
  RegisterHexagonHtaOpenclRuntime(runtime_registry);
#endif  // MACE_ENABLE_OPENCL
#endif  // MACE_ENABLE_HTA

#ifdef MACE_ENABLE_APU
  RegisterApuRuntime(runtime_registry);
#endif  // MACE_ENABLE_APU
}

std::unique_ptr<Runtime> SmartCreateRuntime(RuntimeRegistry *runtime_registry,
                                            const RuntimeType runtime_type,
                                            RuntimeContext *runtime_context) {
  RuntimeSubType sub_type = RuntimeSubType::RT_SUB_REF;

#if defined(MACE_ENABLE_RPCMEM) && defined(MACE_ENABLE_OPENCL)
  if (Rpcmem::IsRpcmemSupported()) {
    if (runtime_type == RuntimeType::RT_OPENCL) {
      auto ion_type = OpenclExecutor::FindCurDeviceIonType();
      if (ion_type == IONType::QUALCOMM_ION) {
        sub_type = RuntimeSubType::RT_SUB_QC_ION;
      }
    } else if (runtime_type == RuntimeType::RT_HTA) {
      sub_type = RuntimeSubType::RT_SUB_WITH_OPENCL;
    }
  }
#endif  // MACE_ENABLE_RPCMEM && MACE_ENABLE_OPENCL

  return runtime_registry->CreateRuntime(runtime_type, sub_type,
                                         runtime_context);
}

}  // namespace mace
