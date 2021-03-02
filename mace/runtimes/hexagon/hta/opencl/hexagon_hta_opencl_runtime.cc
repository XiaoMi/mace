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

#include "mace/runtimes/hexagon/hta/opencl/hexagon_hta_opencl_runtime.h"

#include <memory>

#include "mace/core/runtime/runtime_context.h"
#include "mace/core/runtime/runtime_registry.h"
#include "mace/runtimes/hexagon/hta/hexagon_hta_wrapper.h"

namespace mace {

HexagonHtaOpenclRuntime::HexagonHtaOpenclRuntime(
    RuntimeContext *runtime_context) : HexagonHtaRuntime(runtime_context) {}

MaceStatus HexagonHtaOpenclRuntime::Init(const MaceEngineCfgImpl *config_impl,
                                         const MemoryType mem_type) {
  MACE_CHECK(GetRuntimeType() == RuntimeType::RT_HTA);

  if (opencl_ion_runtime_ == nullptr) {
    QcIonRuntimeContext runtime_context(thread_pool_, rpcmem_);
    opencl_ion_runtime_.reset(new OpenclQcIonRuntime(&runtime_context));
    MACE_RETURN_IF_ERROR(opencl_ion_runtime_->Init(config_impl, mem_type));
  }
  hexagon_controller_ = make_unique<HexagonHTAWrapper>(
      opencl_ion_runtime_.get(), opencl_ion_runtime_->GetRpcmem());

  LOG(INFO) << "Hexagon HTA version: 0x" << std::hex
            << hexagon_controller_->GetVersion();

  return HexagonBaseRuntime::Init(config_impl, mem_type);
}

MemoryType HexagonHtaOpenclRuntime::GetBaseMemoryType() {
  return MemoryType::GPU_BUFFER;
}

MaceStatus HexagonHtaOpenclRuntime::MapBuffer(Buffer *buffer,
                                              bool wait_for_finish) {
  return opencl_ion_runtime_->MapBuffer(buffer, wait_for_finish);
}

MaceStatus HexagonHtaOpenclRuntime::UnMapBuffer(Buffer *buffer) {
  return opencl_ion_runtime_->UnMapBuffer(buffer);
}

OpenclQcIonRuntime *HexagonHtaOpenclRuntime::GetOpenclRuntime() {
  return opencl_ion_runtime_.get();
}

std::shared_ptr<Rpcmem> HexagonHtaOpenclRuntime::GetRpcmem() {
  return opencl_ion_runtime_->GetRpcmem();
}

MemoryManager *HexagonHtaOpenclRuntime::GetMemoryManager(
    const MemoryType mem_type) {
  if (mem_type == MemoryType::GPU_BUFFER) {
    return opencl_ion_runtime_->GetMemoryManager(mem_type);
  }
  return HexagonBaseRuntime::GetMemoryManager(mem_type);
}

void RegisterHexagonHtaOpenclRuntime(RuntimeRegistry *runtime_registry) {
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_HTA,
                        RuntimeSubType::RT_SUB_WITH_OPENCL,
                        HexagonHtaOpenclRuntime);
}

}  // namespace mace
