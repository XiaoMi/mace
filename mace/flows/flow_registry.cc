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

#include "mace/core/flow/flow_registry.h"

namespace mace {

extern void RegisterCpuRefFlow(FlowRegistry *flow_registry);

#if defined(MACE_ENABLE_BFLOAT16)
extern void RegisterCpuBf16Flow(FlowRegistry *flow_registry);
#endif  // MACE_ENABLE_BFLOAT16

#if defined(MACE_ENABLE_FP16)
extern void RegisterCpuFp16Flow(FlowRegistry *flow_registry);
#endif  // MACE_ENABLE_FP16

#ifdef MACE_ENABLE_OPENCL
extern void RegisterOpenclRefFlow(FlowRegistry *flow_registry);
#endif  // MACE_ENABLE_OPENCL

#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
extern void RegisterHexagonRefFlow(FlowRegistry *flow_registry);
#endif  // MACE_ENABLE_HEXAGON || MACE_ENABLE_HTA

#ifdef MACE_ENABLE_APU
extern void RegisterApuRefFlow(FlowRegistry *flow_registry);
#endif  // MACE_ENABLE_APU

void RegisterAllFlows(FlowRegistry *flow_registry) {
  RegisterCpuRefFlow(flow_registry);

#if defined(MACE_ENABLE_BFLOAT16)
  RegisterCpuBf16Flow(flow_registry);
#endif  // MACE_ENABLE_BFLOAT16

#if defined(MACE_ENABLE_FP16)
  RegisterCpuFp16Flow(flow_registry);
#endif  // MACE_ENABLE_FP16

#ifdef MACE_ENABLE_OPENCL
  RegisterOpenclRefFlow(flow_registry);
#endif  // MACE_ENABLE_OPENCL

#if defined(MACE_ENABLE_HEXAGON) || defined(MACE_ENABLE_HTA)
  RegisterHexagonRefFlow(flow_registry);
#endif  // MACE_ENABLE_HEXAGON || MACE_ENABLE_HTA

#ifdef MACE_ENABLE_APU
  RegisterApuRefFlow(flow_registry);
#endif  // MACE_ENABLE_APU
}

}  // namespace mace
