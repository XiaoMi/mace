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

#include "mace/runtimes/hexagon/hta/hexagon_hta_runtime.h"

#include <memory>

#include "mace/core/runtime/runtime_registry.h"
#include "mace/runtimes/hexagon/hta/hexagon_hta_wrapper.h"

namespace mace {

HexagonHtaRuntime::HexagonHtaRuntime(RuntimeContext *runtime_context)
    : HexagonBaseRuntime(runtime_context) {}

MaceStatus HexagonHtaRuntime::Init(const MaceEngineCfgImpl *config_impl,
                                   const MemoryType mem_type) {
  MACE_CHECK(GetRuntimeType() == RuntimeType::RT_HTA);
  hexagon_controller_ = make_unique<HexagonHTAWrapper>(this, GetRpcmem());

  LOG(INFO) << "Hexagon HTA version: 0x" << std::hex
            << hexagon_controller_->GetVersion();

  return HexagonBaseRuntime::Init(config_impl, mem_type);
}

RuntimeType HexagonHtaRuntime::GetRuntimeType() {
  return RuntimeType::RT_HTA;
}

void RegisterHexagonHtaRuntime(RuntimeRegistry *runtime_registry) {
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_HTA,
                        RuntimeSubType::RT_SUB_REF, HexagonHtaRuntime);
}

}  // namespace mace
