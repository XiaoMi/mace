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

#include "mace/core/memory/rpcmem/rpcmem.h"

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

#ifdef MACE_ENABLE_MTK_APU
extern void RegisterApuRuntime(RuntimeRegistry *runtime_registry);
#ifdef MACE_ENABLE_RPCMEM
extern void RegisterApuIonRuntime(RuntimeRegistry *runtime_registry);
#endif  // MACE_ENABLE_RPCMEM
#endif  // MACE_ENABLE_MTK_APU

#ifdef MACE_ENABLE_QNN
extern void RegisterQnnRuntime(RuntimeRegistry *runtime_registry);
#ifdef MACE_ENABLE_OPENCL
extern void RegisterQnnOpenclRuntime(RuntimeRegistry *runtime_registry);
#endif  // MACE_ENABLE_OPENCL
#endif  // MACE_ENABLE_QNN

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

#ifdef MACE_ENABLE_MTK_APU
  RegisterApuRuntime(runtime_registry);
#ifdef MACE_ENABLE_RPCMEM
  RegisterApuIonRuntime(runtime_registry);
#endif  // MACE_ENABLE_RPCMEM
#endif  // MACE_ENABLE_MTK_APU

#ifdef MACE_ENABLE_QNN
  RegisterQnnRuntime(runtime_registry);
#ifdef MACE_ENABLE_OPENCL
  RegisterQnnOpenclRuntime(runtime_registry);
#endif  // MACE_ENABLE_OPENCL
#endif  // MACE_ENABLE_QNN
}

RuntimeSubType SmartGetRuntimeSubType(const RuntimeType runtime_type,
                                      RuntimeContext *runtime_context) {
  RuntimeSubType sub_type = RuntimeSubType::RT_SUB_REF;

  MACE_UNUSED(runtime_type);
#ifdef MACE_ENABLE_RPCMEM
  if (runtime_context->context_type == RCT_ION) {
    auto ion_rct = static_cast<IonRuntimeContext *>(runtime_context);
    if (ion_rct->rpcmem && ion_rct->rpcmem->IsRpcmemSupported()) {
#ifdef MACE_ENABLE_MTK_APU
      if (runtime_type == RuntimeType::RT_APU) {
        sub_type = RuntimeSubType::RT_SUB_ION;
      }
#endif  // MACE_ENABLE_MTK_APU
#ifdef MACE_ENABLE_OPENCL
      if (runtime_type == RuntimeType::RT_OPENCL ||
          runtime_type == RuntimeType::RT_CPU) {
        auto ion_type = OpenclExecutor::FindCurDeviceIonType();
        if (ion_type == IONType::QUALCOMM_ION) {
          sub_type = RuntimeSubType::RT_SUB_ION;
        }
      } else if (runtime_type == RuntimeType::RT_HTA) {
        sub_type = RuntimeSubType::RT_SUB_WITH_OPENCL;
      } else if (runtime_type == RuntimeType::RT_HTP) {
        sub_type = RuntimeSubType::RT_SUB_WITH_OPENCL;
      }
#endif  // MACE_ENABLE_OPENCL
    }
  }
#else
  MACE_UNUSED(runtime_context);
#endif  // MACE_ENABLE_RPCMEM
  return sub_type;
}

std::unique_ptr<Runtime> SmartCreateRuntime(RuntimeRegistry *runtime_registry,
                                            const RuntimeType runtime_type,
                                            RuntimeContext *runtime_context) {
  auto sub_type = SmartGetRuntimeSubType(runtime_type, runtime_context);
  return runtime_registry->CreateRuntime(runtime_type, sub_type,
                                         runtime_context);
}

}  // namespace mace
