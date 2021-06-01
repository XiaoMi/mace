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

#ifndef MACE_CORE_OPS_OP_INIT_CONTEXT_H_
#define MACE_CORE_OPS_OP_INIT_CONTEXT_H_

#include "mace/public/mace.h"

namespace mace {
class Workspace;
class Runtime;

// memory_optimizer, device
class OpInitContext {
 public:
  explicit OpInitContext(Workspace *ws, Runtime *runtime = nullptr,
                         Runtime *cpu_runtime = nullptr);
  ~OpInitContext() = default;

  Workspace *workspace() const;

  void SetRuntime(Runtime *runtime);

  void SetCpuRuntime(Runtime *cpu_runtime);

  Runtime *runtime() const;

  Runtime *GetRuntimeByMemType(MemoryType mem_type) const;

 private:
  Workspace *ws_;
  Runtime *runtime_;
  Runtime *cpu_runtime_;
};

}  // namespace mace

#endif  // MACE_CORE_OPS_OP_INIT_CONTEXT_H_
