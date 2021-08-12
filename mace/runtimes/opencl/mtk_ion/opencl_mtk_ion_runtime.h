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

#ifndef MACE_RUNTIMES_OPENCL_MTK_ION_OPENCL_MTK_ION_RUNTIME_H_
#define MACE_RUNTIMES_OPENCL_MTK_ION_OPENCL_MTK_ION_RUNTIME_H_

#include <memory>

#include "mace/core/runtime/runtime_context.h"
#include "mace/runtimes/opencl/mtk_ion/opencl_buffer_mtk_ion_allocator.h"
#include "mace/runtimes/opencl/mtk_ion/opencl_image_mtk_ion_allocator.h"
#include "mace/runtimes/opencl/opencl_image_manager.h"
#include "mace/runtimes/opencl/opencl_runtime.h"

namespace mace {

class OpContext;

class OpenclMtkIonRuntime : public OpenclRuntime {
 public:
  explicit OpenclMtkIonRuntime(RuntimeContext *runtime_context);
  virtual ~OpenclMtkIonRuntime() = default;

  MaceStatus Init(const MaceEngineCfgImpl *engine_config,
                  const MemoryType mem_type) override;
  RuntimeSubType GetRuntimeSubType() override;
  MaceStatus MapBuffer(Buffer *buffer, bool wait_for_finish) override;
  MaceStatus UnMapBuffer(Buffer *buffer) override;

  MemoryManager *GetMemoryManager(MemoryType mem_type) override;

  std::shared_ptr<Rpcmem> GetRpcmem();

 protected:
  MaceStatus CreateOpenclExecutorAndInit(
      const MaceEngineCfgImpl *engine_config) override;

 private:
  std::shared_ptr<Rpcmem> rpcmem_;
  std::shared_ptr<OpenclBufferMtkIonAllocator> buffer_ion_allocator_;
  std::shared_ptr<OpenclImageMtkIonAllocator> image_ion_allocator_;
  std::unique_ptr<GeneralMemoryManager> buffer_ion_manager_;
  std::unique_ptr<OpenclImageManager> image_ion_manager_;
};

}  // namespace mace

#endif  // MACE_RUNTIMES_OPENCL_MTK_ION_OPENCL_MTK_ION_RUNTIME_H_
