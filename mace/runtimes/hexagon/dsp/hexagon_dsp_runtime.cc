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

#include "mace/runtimes/hexagon/dsp/hexagon_dsp_runtime.h"

#include <memory>

#include "mace/core/runtime/runtime_registry.h"
#include "mace/public/mace.h"
#include "mace/runtimes/hexagon/dsp/hexagon_dsp_wrapper.h"

namespace mace {

HexagonDspRuntime::HexagonDspRuntime(RuntimeContext *runtime_context)
    : HexagonBaseRuntime(runtime_context) {}

MaceStatus HexagonDspRuntime::Init(const MaceEngineCfgImpl *config_impl,
                                   const MemoryType mem_type) {
  MACE_UNUSED(mem_type);
  MACE_CHECK(GetRuntimeType() == RuntimeType::RT_HEXAGON);
  HexagonDSPWrapper::SetPower(config_impl->hexagon_corner(),
                              config_impl->hexagon_dcvs_enable(),
                              config_impl->hexagon_latency());
  VLOG(1) << "config_impl->hexagon_corner(): "
          << config_impl->hexagon_corner()
          << ", hexagon_dcvs_enable: " << config_impl->hexagon_dcvs_enable()
          << ", hexagon_latency: " << config_impl->hexagon_latency();

  hexagon_controller_ = make_unique<HexagonDSPWrapper>();
  LOG(INFO) << "Hexagon DSP version: 0x" << std::hex
            << hexagon_controller_->GetVersion();

  return HexagonBaseRuntime::Init(config_impl, mem_type);
}

RuntimeType HexagonDspRuntime::GetRuntimeType() {
  return RuntimeType::RT_HEXAGON;
}

void RegisterHexagonDspRuntime(RuntimeRegistry *runtime_registry) {
  MACE_REGISTER_RUNTIME(runtime_registry, RuntimeType::RT_HEXAGON,
                        RuntimeSubType::RT_SUB_REF, HexagonDspRuntime);
}

}  // namespace mace
