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

#include "mace/runtimes/opencl/qc_ion/opencl_qc_ion_runtime.h"

#include <memory>

#include "mace/core/runtime/runtime_registry.h"
#include "mace/runtimes/opencl/core/opencl_context.h"
#include "mace/runtimes/opencl/qc_ion/opencl_base_qc_ion_allocator.h"
#include "mace/runtimes/opencl/qc_ion/opencl_qc_ion_executor.h"

namespace mace {

OpenclQcIonRuntime::OpenclQcIonRuntime(RuntimeContext *runtime_context)
    : OpenclRuntime(runtime_context) {
  MACE_CHECK(runtime_context->context_type == RuntimeContextType::RCT_QC_ION);
  QcIonRuntimeContext *qc_ion_runtime_context =
      static_cast<QcIonRuntimeContext *>(runtime_context);
  rpcmem_ = qc_ion_runtime_context->rpcmem;
}

MaceStatus OpenclQcIonRuntime::Init(const MaceEngineCfgImpl *engine_config,
                                    const MemoryType mem_type) {
  MACE_RETURN_IF_ERROR(OpenclRuntime::Init(engine_config, mem_type));

  buffer_ion_allocator_ =
      make_unique<OpenclBufferQcIonAllocator>(opencl_executor_.get(), rpcmem_);
  image_ion_allocator_ =
      make_unique<OpenclImageQcIonAllocator>(opencl_executor_.get(), rpcmem_);
  buffer_ion_manager_ =
      make_unique<GeneralMemoryManager>(buffer_ion_allocator_.get());
  image_ion_manager_ =
      make_unique<OpenclImageManager>(image_ion_allocator_.get());

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpenclQcIonRuntime::CreateOpenclExecutorAndInit(
    const MaceEngineCfgImpl *engine_config) {
  if (opencl_executor_ == nullptr) {
    opencl_executor_ = make_unique<OpenclQcIonExecutor>(
        engine_config->opencl_context()->opencl_cache_storage(),
        engine_config->opencl_context()->opencl_binary_storage(),
        engine_config->opencl_context()->opencl_tuner());
    MACE_RETURN_IF_ERROR(opencl_executor_->Init(
        engine_config->gpu_priority_hint(), engine_config->gpu_perf_hint()));
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpenclQcIonRuntime::MapBuffer(Buffer *buffer, bool wait_for_finish) {
  auto *opencl_executor = OpenclQcIonExecutor::Get(opencl_executor_.get());
  MACE_CHECK((buffer->mem_type == MemoryType::GPU_BUFFER ||
      buffer->mem_type == MemoryType::GPU_IMAGE) &&
      opencl_executor->ion_type() == IONType::QUALCOMM_ION);
  MACE_LATENCY_LOGGER(1, "OpenclQcIonRuntime Map OpenCL buffer");

  OpenclBaseQcIonAllocator *ion_allocator = nullptr;
  if (buffer->mem_type == MemoryType::GPU_IMAGE) {
    ion_allocator = image_ion_allocator_.get();
  } else {
    ion_allocator = buffer_ion_allocator_.get();
  }

  void *mapped_ptr =
      ion_allocator->GetMappedPtrByIonBuffer(buffer->mutable_memory<void>());
  MACE_CHECK(mapped_ptr != nullptr, "Try to map unallocated Buffer!");

  if (wait_for_finish) {
    opencl_executor->command_queue().finish();
  }

  auto policy = opencl_executor->qcom_host_cache_policy();
  if (policy == CL_MEM_HOST_WRITEBACK_QCOM) {
    const auto ret = rpcmem_->SyncCacheStart(mapped_ptr);
    MACE_CHECK(ret == 0, "Ion map failed, ret = ", ret);
  }

  buffer->SetHost(static_cast<uint8_t *>(mapped_ptr) + buffer->offset());
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpenclQcIonRuntime::UnMapBuffer(Buffer *buffer) {
  auto *opencl_executor = OpenclQcIonExecutor::Get(opencl_executor_.get());
  MACE_CHECK(opencl_executor->ion_type() == IONType::QUALCOMM_ION);
  MACE_LATENCY_LOGGER(1, "OpenclQcIonRuntime Unmap OpenCL buffer/Image");

  const auto cache_policy = opencl_executor->qcom_host_cache_policy();
  if (cache_policy == CL_MEM_HOST_WRITEBACK_QCOM &&
      buffer->data<void>() != nullptr) {
    auto *mapped_ptr = buffer->mutable_data<uint8_t>() - buffer->offset();
    MACE_CHECK(rpcmem_->SyncCacheEnd(mapped_ptr) == 0);
  }
  buffer->SetHost(nullptr);
  return MaceStatus::MACE_SUCCESS;
}

MemoryManager *OpenclQcIonRuntime::GetMemoryManager(MemoryType mem_type) {
  MemoryManager *buffer_manager = nullptr;
  if (mem_type == MemoryType::GPU_BUFFER) {
    buffer_manager = buffer_ion_manager_.get();
  } else if (mem_type == MemoryType::GPU_IMAGE) {
    buffer_manager = image_ion_manager_.get();
  } else {
    MACE_CHECK(false, "OpenclRuntime::GetMemoryManagerByMemType",
               "find an invalid mem type:", mem_type);
  }

  return buffer_manager;
}

std::shared_ptr<Rpcmem> OpenclQcIonRuntime::GetRpcmem() {
  return rpcmem_;
}

void RegisterOpenclQcIonRuntime(RuntimeRegistry *runtime_registry) {
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_OPENCL,
                        RuntimeSubType::RT_SUB_QC_ION, OpenclQcIonRuntime);
}

}  // namespace mace
