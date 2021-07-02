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

#include "mace/runtimes/qnn/qnn_base_runtime.h"

#include "mace/core/runtime/runtime_context.h"
#include "mace/core/runtime/runtime_registry.h"
#include "mace/public/mace.h"

namespace mace {

QnnBaseRuntime::QnnBaseRuntime(RuntimeContext *runtime_context)
    : Runtime(runtime_context) {
  MACE_CHECK(runtime_context->context_type == RuntimeContextType::RCT_ION);
  IonRuntimeContext *ion_runtime_context =
      static_cast<IonRuntimeContext *>(runtime_context);
  rpcmem_ = ion_runtime_context->rpcmem;
  ion_allocator_ = make_unique<CpuIonAllocator>(rpcmem_);
  buffer_manager_ = make_unique<GeneralMemoryManager>(ion_allocator_.get());
}

QnnBaseRuntime::~QnnBaseRuntime() {
  MACE_CHECK(qnn_wrapper_->Destroy(), "Qnn destroy error.");
}

QnnBaseRuntime *QnnBaseRuntime::Get(Runtime *runtime) {
  return static_cast<QnnBaseRuntime *>(runtime);
}

MaceStatus QnnBaseRuntime::Init(const MaceEngineCfgImpl *config_impl,
                            const MemoryType mem_type) {
  MACE_UNUSED(mem_type);
  cache_policy_ = config_impl->accelerator_cache_policy();
  cache_binary_file_ = config_impl->accelerator_binary_file();
  cache_storage_file_ = config_impl->accelerator_storage_file();
  perf_type_ = config_impl->hexagon_performance();
  return MaceStatus::MACE_SUCCESS;
}

std::unique_ptr<Buffer> QnnBaseRuntime::MakeSliceBuffer(
    const NetDef &net_def, const unsigned char *model_data,
    const index_t model_data_size) {
  MACE_UNUSED(net_def);
  MACE_UNUSED(model_data);
  MACE_UNUSED(model_data_size);
  return nullptr;
}

QnnWrapper *QnnBaseRuntime::GetQnnWrapper() {
  return qnn_wrapper_.get();
}

std::shared_ptr<Rpcmem> QnnBaseRuntime::GetRpcmem() {
  return ion_allocator_->GetRpcmem();
}

MemoryManager *QnnBaseRuntime::GetMemoryManager(const MemoryType mem_type) {
  MemoryManager *buffer_manager = nullptr;
  if (mem_type == MemoryType::CPU_BUFFER) {
    buffer_manager = buffer_manager_.get();
  } else {
    MACE_CHECK(false, "QnnBaseRuntime::GetMemoryManagerByMemType",
               "find an invalid mem type:", mem_type);
  }

  return buffer_manager;
}

RuntimeType QnnBaseRuntime::GetRuntimeType() {
  return RuntimeType::RT_HTP;
}

AcceleratorCachePolicy QnnBaseRuntime::GetCachePolicy() {
  return cache_policy_;
}

std::string QnnBaseRuntime::GetCacheStorePath() {
  return cache_storage_file_;
}

std::string QnnBaseRuntime::GetCacheLoadPath() {
  return cache_binary_file_;
}

}  // namespace mace
