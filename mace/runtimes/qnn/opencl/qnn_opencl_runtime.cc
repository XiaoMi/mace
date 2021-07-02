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

#include "mace/runtimes/qnn/opencl/qnn_opencl_runtime.h"

#include <memory>

#include "mace/core/runtime/runtime_context.h"
#include "mace/core/runtime/runtime_registry.h"

namespace mace {

QnnOpenclRuntime::QnnOpenclRuntime(
    RuntimeContext *runtime_context) : QnnRuntime(runtime_context) {}

MaceStatus QnnOpenclRuntime::Init(const MaceEngineCfgImpl *config_impl,
                                         const MemoryType mem_type) {
  MACE_CHECK(GetRuntimeType() == RuntimeType::RT_HTP);

  if (opencl_ion_runtime_ == nullptr) {
    IonRuntimeContext runtime_context(thread_pool_, rpcmem_);
    opencl_ion_runtime_.reset(new OpenclQcIonRuntime(&runtime_context));
    MACE_RETURN_IF_ERROR(opencl_ion_runtime_->Init(config_impl, mem_type));
  }
  qnn_wrapper_ = make_unique<QnnWrapper>(opencl_ion_runtime_.get());

  return QnnBaseRuntime::Init(config_impl, mem_type);
}

RuntimeSubType QnnOpenclRuntime::GetRuntimeSubType() {
  return RuntimeSubType::RT_SUB_WITH_OPENCL;
}

MemoryType QnnOpenclRuntime::GetBaseMemoryType() {
  return MemoryType::GPU_BUFFER;
}

MaceStatus QnnOpenclRuntime::MapBuffer(Buffer *buffer,
                                              bool wait_for_finish) {
  return opencl_ion_runtime_->MapBuffer(buffer, wait_for_finish);
}

MaceStatus QnnOpenclRuntime::UnMapBuffer(Buffer *buffer) {
  return opencl_ion_runtime_->UnMapBuffer(buffer);
}

OpenclQcIonRuntime *QnnOpenclRuntime::GetOpenclRuntime() {
  return opencl_ion_runtime_.get();
}

std::shared_ptr<Rpcmem> QnnOpenclRuntime::GetRpcmem() {
  return opencl_ion_runtime_->GetRpcmem();
}

MemoryManager *QnnOpenclRuntime::GetMemoryManager(
    const MemoryType mem_type) {
  if (mem_type == MemoryType::GPU_BUFFER) {
    return opencl_ion_runtime_->GetMemoryManager(mem_type);
  }
  return QnnRuntime::GetMemoryManager(mem_type);
}

void RegisterQnnOpenclRuntime(RuntimeRegistry *runtime_registry) {
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_HTP,
                        RuntimeSubType::RT_SUB_WITH_OPENCL,
                        QnnOpenclRuntime);
}
}  // namespace mace
