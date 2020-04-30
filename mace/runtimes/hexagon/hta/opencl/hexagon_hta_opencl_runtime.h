// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_RUNTIMES_HEXAGON_HTA_OPENCL_HEXAGON_HTA_OPENCL_RUNTIME_H_
#define MACE_RUNTIMES_HEXAGON_HTA_OPENCL_HEXAGON_HTA_OPENCL_RUNTIME_H_

#include <memory>

#include "mace/runtimes/hexagon/hta/hexagon_hta_runtime.h"
#include "mace/runtimes/opencl/qc_ion/opencl_qc_ion_runtime.h"

namespace mace {

class HexagonHtaOpenclRuntime : public HexagonHtaRuntime {
 public:
  explicit HexagonHtaOpenclRuntime(RuntimeContext *runtime_context);
  virtual ~HexagonHtaOpenclRuntime() = default;

  MaceStatus Init(const MaceEngineCfgImpl *engine_config,
                  const MemoryType mem_type) override;

  MemoryType GetBaseMemoryType() override;
  MaceStatus MapBuffer(Buffer *buffer, bool wait_for_finish) override;
  MaceStatus UnMapBuffer(Buffer *buffer) override;
  OpenclQcIonRuntime *GetOpenclRuntime();
  std::shared_ptr<Rpcmem> GetRpcmem() override;

 protected:
  MemoryManager *GetMemoryManager(const MemoryType mem_type) override;

 private:
  // TODO(luxuhui): replace it by subgraph
  std::unique_ptr<OpenclQcIonRuntime> opencl_ion_runtime_;
};
}  // namespace mace

#endif  // MACE_RUNTIMES_HEXAGON_HTA_OPENCL_HEXAGON_HTA_OPENCL_RUNTIME_H_

