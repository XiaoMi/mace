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


#include "mace/core/ops/op_init_context.h"
#include "mace/core/runtime/runtime.h"

namespace mace {

OpInitContext::OpInitContext(Workspace *ws, Runtime *runtime,
                             Runtime *cpu_runtime)
    : ws_(ws), runtime_(runtime), cpu_runtime_(cpu_runtime) {}

Workspace *OpInitContext::workspace() const {
  return ws_;
}

void OpInitContext::SetRuntime(Runtime *runtime) {
  runtime_ = runtime;
}

void OpInitContext::SetCpuRuntime(Runtime *cpu_runtime) {
  cpu_runtime_ = cpu_runtime;
}

Runtime *OpInitContext::runtime() const {
  return runtime_;
}


Runtime *OpInitContext::GetRuntimeByMemType(MemoryType mem_type) const {
  if (mem_type == GPU_IMAGE || mem_type == GPU_BUFFER) {
    MACE_CHECK(runtime_->GetRuntimeType() == RT_OPENCL);
    return runtime_;
  } else if (mem_type == CPU_BUFFER) {
    return cpu_runtime_;
  } else {
    MACE_NOT_IMPLEMENTED;
    return nullptr;
  }
}


}  // namespace mace
