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

#include "mace/runtimes/hexagon/hexagon_base_runtime.h"

#include "mace/core/runtime/runtime_context.h"
#include "mace/public/mace.h"

namespace mace {

HexagonBaseRuntime::HexagonBaseRuntime(RuntimeContext *runtime_context)
    : Runtime(runtime_context) {
  MACE_CHECK(runtime_context->context_type == RuntimeContextType::RCT_QC_ION);
  QcIonRuntimeContext *qc_ion_runtime_context =
      static_cast<QcIonRuntimeContext *>(runtime_context);
  rpcmem_ = qc_ion_runtime_context->rpcmem;
  ion_allocator_ = make_unique<CpuIonAllocator>(rpcmem_);
  buffer_manager_ = make_unique<GeneralMemoryManager>(ion_allocator_.get());
}

HexagonBaseRuntime::~HexagonBaseRuntime() {
  if (VLOG_IS_ON(2)) {
    hexagon_controller_->PrintLog();
  }
  if (VLOG_IS_ON(1)) {
    hexagon_controller_->GetPerfInfo();
  }
  MACE_CHECK(hexagon_controller_->TeardownGraph(), "hexagon teardown error");
  MACE_CHECK(hexagon_controller_->Finalize(), "hexagon finalize error");
}

HexagonBaseRuntime *HexagonBaseRuntime::Get(Runtime *runtime) {
  return static_cast<HexagonBaseRuntime *>(runtime);
}

MaceStatus HexagonBaseRuntime::Init(const MaceEngineCfgImpl *config_impl,
                                    const MemoryType mem_type) {
  MACE_UNUSED(config_impl);
  MACE_UNUSED(mem_type);
  MACE_CHECK(hexagon_controller_->Config(), "hexagon config error");
  MACE_CHECK(hexagon_controller_->Init(), "hexagon init error");
  hexagon_controller_->SetDebugLevel(
      static_cast<int>(mace::port::MinVLogLevelFromEnv()));

  return MaceStatus::MACE_SUCCESS;
}

std::unique_ptr<Buffer> HexagonBaseRuntime::MakeSliceBuffer(
    const NetDef &net_def, const unsigned char *model_data,
    const index_t model_data_size) {
  MACE_UNUSED(net_def);
  MACE_UNUSED(model_data);
  MACE_UNUSED(model_data_size);
  return nullptr;
}

bool HexagonBaseRuntime::ExecuteGraphNew(
    const std::map<std::string, Tensor *> &input_tensors,
    std::map<std::string, Tensor *> *output_tensors) {
  return hexagon_controller_->ExecuteGraphNew(input_tensors, output_tensors);
}

HexagonControlWrapper *HexagonBaseRuntime::GetHexagonController() {
  return hexagon_controller_.get();
}

MemoryManager *HexagonBaseRuntime::GetMemoryManager(const MemoryType mem_type) {
  MemoryManager *buffer_manager = nullptr;
  if (mem_type == MemoryType::CPU_BUFFER) {
    buffer_manager = buffer_manager_.get();
  } else {
    MACE_CHECK(false, "HexagonBaseRuntime::GetMemoryManagerByMemType",
               "find an invalid mem type:", mem_type);
  }

  return buffer_manager;
}

std::shared_ptr<Rpcmem> HexagonBaseRuntime::GetRpcmem() {
  return ion_allocator_->GetRpcmem();
}

}  // namespace mace
