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

#ifndef MACE_RUNTIMES_OPENCL_OPENCL_REF_RUNTIME_H_
#define MACE_RUNTIMES_OPENCL_OPENCL_REF_RUNTIME_H_

#include <memory>

#include "mace/runtimes/opencl/opencl_buffer_allocator.h"
#include "mace/runtimes/opencl/opencl_image_allocator.h"
#include "mace/runtimes/opencl/opencl_image_manager.h"
#include "mace/runtimes/opencl/opencl_runtime.h"

namespace mace {

class OpContext;

class OpenclRefRuntime : public OpenclRuntime {
 public:
  explicit OpenclRefRuntime(RuntimeContext *runtime_context);

  MaceStatus Init(const MaceEngineCfgImpl *engine_config,
                  const MemoryType mem_type) override;
  MaceStatus MapBuffer(Buffer *buffer, bool wait_for_finish) override;
  MaceStatus UnMapBuffer(Buffer *buffer) override;

 protected:
  MemoryManager *GetMemoryManager(MemoryType mem_type) override;

 private:
  std::shared_ptr<OpenclBufferAllocator> buffer_allocator_;
  std::shared_ptr<OpenclImageAllocator> image_allocator_;
  std::unique_ptr<GeneralMemoryManager> buffer_manager_;
  std::unique_ptr<OpenclImageManager> image_manager_;
};

}  // namespace mace

#endif  // MACE_RUNTIMES_OPENCL_OPENCL_REF_RUNTIME_H_
