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

#include "mace/runtimes/opencl/mtk_ion/opencl_mtk_ion_runtime.h"

#include <memory>

#include "mace/core/runtime/runtime_registry.h"
#include "mace/runtimes/opencl/core/opencl_context.h"
#include "mace/runtimes/opencl/mtk_ion/opencl_base_mtk_ion_allocator.h"
#include "mace/runtimes/opencl/mtk_ion/opencl_mtk_ion_executor.h"

namespace mace {

OpenclMtkIonRuntime::OpenclMtkIonRuntime(RuntimeContext *runtime_context)
    : OpenclRuntime(runtime_context) {
  MACE_CHECK(runtime_context->context_type == RuntimeContextType::RCT_ION);
  auto *ion_runtime_context = static_cast<IonRuntimeContext *>(runtime_context);
  rpcmem_ = ion_runtime_context->rpcmem;
}

MaceStatus OpenclMtkIonRuntime::Init(const MaceEngineCfgImpl *engine_config,
                                    const MemoryType mem_type) {
  MACE_RETURN_IF_ERROR(OpenclRuntime::Init(engine_config, mem_type));

  buffer_ion_allocator_ =
      make_unique<OpenclBufferMtkIonAllocator>(opencl_executor_.get(), rpcmem_);
  image_ion_allocator_ =
      make_unique<OpenclImageMtkIonAllocator>(opencl_executor_.get(), rpcmem_);
  buffer_ion_manager_ =
      make_unique<GeneralMemoryManager>(buffer_ion_allocator_.get());
  image_ion_manager_ =
      make_unique<OpenclImageManager>(image_ion_allocator_.get());

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpenclMtkIonRuntime::CreateOpenclExecutorAndInit(
    const MaceEngineCfgImpl *engine_config) {
  if (opencl_executor_ == nullptr) {
    opencl_executor_ = make_unique<OpenclMtkIonExecutor>();
    MACE_RETURN_IF_ERROR(opencl_executor_->Init(
        engine_config->opencl_context(), engine_config->gpu_priority_hint(),
        engine_config->gpu_perf_hint()));
  }
  return MaceStatus::MACE_SUCCESS;
}

RuntimeSubType OpenclMtkIonRuntime::GetRuntimeSubType() {
  return RuntimeSubType::RT_SUB_MTK_ION;
}

MaceStatus OpenclMtkIonRuntime::MapBuffer(Buffer *buffer, bool wait_for_finish) {
  auto *opencl_executor = OpenclMtkIonExecutor::Get(opencl_executor_.get());
  MACE_CHECK((buffer->mem_type == MemoryType::GPU_BUFFER ||
      buffer->mem_type == MemoryType::GPU_IMAGE) &&
      opencl_executor->ion_type() == IONType::MTK_ION);
  MACE_LATENCY_LOGGER(1, "OpenclMtkIonRuntime Map OpenCL buffer");

  OpenclBaseMtkIonAllocator *ion_allocator = nullptr;
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


  const auto ret = rpcmem_->SyncCacheStart(mapped_ptr);
  MACE_CHECK(ret == 0, "Ion map failed, ret = ", ret);

  buffer->SetHost(static_cast<uint8_t *>(mapped_ptr) + buffer->offset());
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpenclMtkIonRuntime::UnMapBuffer(Buffer *buffer) {
  auto *opencl_executor = OpenclMtkIonExecutor::Get(opencl_executor_.get());
  MACE_CHECK(opencl_executor->ion_type() == IONType::MTK_ION);
  MACE_LATENCY_LOGGER(1, "OpenclMtkIonRuntime Unmap OpenCL buffer/Image");

  if (buffer->data<void>() != nullptr) {
    auto *mapped_ptr = buffer->mutable_data<uint8_t>() - buffer->offset();
    MACE_CHECK(rpcmem_->SyncCacheEnd(mapped_ptr) == 0);
  }
  buffer->SetHost(nullptr);
  return MaceStatus::MACE_SUCCESS;
}

MemoryManager *OpenclMtkIonRuntime::GetMemoryManager(MemoryType mem_type) {
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

std::shared_ptr<Rpcmem> OpenclMtkIonRuntime::GetRpcmem() {
  return rpcmem_;
}

void RegisterOpenclMtkIonRuntime(RuntimeRegistry *runtime_registry) {
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_OPENCL,
                        RuntimeSubType::RT_SUB_MTK_ION, OpenclMtkIonRuntime);
}

}  // namespace mace
