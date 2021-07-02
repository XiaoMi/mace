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

#include "mace/runtimes/apu/apu_runtime.h"

#include "mace/core/runtime/runtime_registry.h"
#include "mace/runtimes/cpu/cpu_ref_allocator.h"

namespace mace {

ApuRuntime::ApuRuntime(RuntimeContext *runtime_context)
    : Runtime(runtime_context),
      apu_wrapper_(make_unique<ApuWrapper>(this)) {}

ApuRuntime::~ApuRuntime() {
  MACE_CHECK(apu_wrapper_->Uninit(), "apu uninit error");
}

ApuRuntime *ApuRuntime::Get(Runtime *runtime) {
  return static_cast<ApuRuntime *>(runtime);
}

std::unique_ptr<Allocator> ApuRuntime::CreateAllocator() {
  return make_unique<CpuRefAllocator>();
}

MaceStatus ApuRuntime::Init(const MaceEngineCfgImpl *engine_config,
                            const MemoryType mem_type) {
  MACE_UNUSED(mem_type);
  apu_cache_policy_ = engine_config->accelerator_cache_policy();
  apu_binary_file_ = engine_config->accelerator_binary_file();
  apu_storage_file_ = engine_config->accelerator_storage_file();
  apu_boost_hint_ = engine_config->apu_boost_hint();
  apu_preference_hint_ = engine_config->apu_preference_hint();

  allocator_ = CreateAllocator();
  buffer_manager_ = make_unique<GeneralMemoryManager>(allocator_.get());

  return MaceStatus::MACE_SUCCESS;
}

std::unique_ptr<Buffer> ApuRuntime::MakeSliceBuffer(
    const NetDef &net_def, const unsigned char *model_data,
    const index_t model_data_size) {
  MACE_UNUSED(net_def);
  MACE_UNUSED(model_data);
  MACE_UNUSED(model_data_size);
  return nullptr;
}

RuntimeType ApuRuntime::GetRuntimeType() {
  return RuntimeType::RT_APU;
}

ApuWrapper *ApuRuntime::GetApuWrapper() {
  return apu_wrapper_.get();
}

AcceleratorCachePolicy ApuRuntime::GetCachePolicy() {
  return apu_cache_policy_;
}

uint8_t ApuRuntime::GetBoostHint() {
  return apu_boost_hint_;
}

APUPreferenceHint ApuRuntime::GetPreferenceHint() {
  return apu_preference_hint_;
}

const char *ApuRuntime::GetCacheStorePath() {
  return apu_storage_file_.c_str();
}

const char *ApuRuntime::GetCacheLoadPath() {
  return apu_binary_file_.c_str();
}

MemoryManager *ApuRuntime::GetMemoryManager(const MemoryType mem_type) {
  MemoryManager *buffer_manager = nullptr;
  if (mem_type == MemoryType::CPU_BUFFER) {
    buffer_manager = buffer_manager_.get();
  } else {
    MACE_CHECK(false, "ApuRuntime::GetMemoryManagerByMemType",
               "find an invalid mem type:", mem_type);
  }

  return buffer_manager;
}

void RegisterApuRuntime(RuntimeRegistry *runtime_registry) {
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_APU,
                        RuntimeSubType::RT_SUB_REF, ApuRuntime);
}

}  // namespace mace
