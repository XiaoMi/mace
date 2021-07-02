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

#ifndef MACE_RUNTIMES_QNN_QNN_BASE_RUNTIME_H_
#define MACE_RUNTIMES_QNN_QNN_BASE_RUNTIME_H_

#include <memory>
#include <string>

#include "mace/core/memory/general_memory_manager.h"
#include "mace/core/runtime/runtime_context.h"
#include "mace/core/runtime/runtime.h"
#include "mace/runtimes/cpu/ion/cpu_ion_allocator.h"
#include "mace/runtimes/qnn/qnn_wrapper.h"

namespace mace {

class QnnBaseRuntime : public Runtime {
 public:
  explicit QnnBaseRuntime(RuntimeContext *runtime_context);
  virtual ~QnnBaseRuntime();
  static QnnBaseRuntime *Get(Runtime *runtime);

  MaceStatus Init(const MaceEngineCfgImpl *engine_config,
                  const MemoryType mem_type) override;

  std::unique_ptr<Buffer> MakeSliceBuffer(
      const NetDef &net_def, const unsigned char *model_data,
      const index_t model_data_size) override;

  MemoryManager *GetMemoryManager(const MemoryType mem_type) override;

  QnnWrapper *GetQnnWrapper();
  virtual std::shared_ptr<Rpcmem> GetRpcmem();
  RuntimeType GetRuntimeType() override;
  AcceleratorCachePolicy GetCachePolicy();
  std::string GetCacheStorePath();
  std::string GetCacheLoadPath();

 protected:
  std::unique_ptr<CpuIonAllocator> ion_allocator_;
  std::unique_ptr<GeneralMemoryManager> buffer_manager_;
  std::shared_ptr<Rpcmem> rpcmem_;
  std::unique_ptr<QnnWrapper> qnn_wrapper_;
  HexagonPerformanceType perf_type_;
  AcceleratorCachePolicy cache_policy_;
  std::string cache_binary_file_;
  std::string cache_storage_file_;
};

}  // namespace mace
#endif  // MACE_RUNTIMES_QNN_QNN_BASE_RUNTIME_H_
