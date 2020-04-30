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

#ifndef MACE_RUNTIMES_HEXAGON_HEXAGON_BASE_RUNTIME_H_
#define MACE_RUNTIMES_HEXAGON_HEXAGON_BASE_RUNTIME_H_

#include <map>
#include <memory>
#include <utility>
#include <string>

#include "mace/core/memory/general_memory_manager.h"
#include "mace/core/runtime/runtime_context.h"
#include "mace/core/runtime/runtime.h"
#include "mace/runtimes/cpu/ion/cpu_ion_allocator.h"
#include "mace/runtimes/hexagon/hexagon_control_wrapper.h"

namespace mace {

class HexagonBaseRuntime : public Runtime {
 public:
  explicit HexagonBaseRuntime(RuntimeContext *runtime_context);
  virtual ~HexagonBaseRuntime();
  static HexagonBaseRuntime *Get(Runtime *runtime);

  MaceStatus Init(const MaceEngineCfgImpl *engine_config,
                  const MemoryType mem_type) override;

  std::unique_ptr<Buffer> MakeSliceBuffer(
      const NetDef &net_def, const unsigned char *model_data,
      const index_t model_data_size) override;

  MemoryManager *GetMemoryManager(const MemoryType mem_type) override;

  bool ExecuteGraphNew(const std::map<std::string, Tensor *> &input_tensors,
                       std::map<std::string, Tensor *> *output_tensors);
  HexagonControlWrapper *GetHexagonController();
  virtual std::shared_ptr<Rpcmem> GetRpcmem();

 protected:
  std::shared_ptr<Rpcmem> rpcmem_;
  std::unique_ptr<HexagonControlWrapper> hexagon_controller_;
  std::unique_ptr<CpuIonAllocator> ion_allocator_;
  std::unique_ptr<GeneralMemoryManager> buffer_manager_;
};

}  // namespace mace
#endif  // MACE_RUNTIMES_HEXAGON_HEXAGON_BASE_RUNTIME_H_
