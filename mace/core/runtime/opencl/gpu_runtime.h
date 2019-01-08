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

#ifndef MACE_CORE_RUNTIME_OPENCL_GPU_RUNTIME_H_
#define MACE_CORE_RUNTIME_OPENCL_GPU_RUNTIME_H_

#include <memory>

#include "mace/proto/mace.pb.h"

namespace mace {

class OpenCLRuntime;
class ScratchImageManager;

class GPURuntime {
 public:
  explicit GPURuntime(OpenCLRuntime *runtime);
  ~GPURuntime();
  OpenCLRuntime *opencl_runtime();
  ScratchImageManager *scratch_image_manager() const;

  // TODO(liuqi): remove this function in the future, make decision at runtime.
  bool UseImageMemory();
  void set_mem_type(MemoryType type);

 private:
  OpenCLRuntime *runtime_;
  std::unique_ptr<ScratchImageManager> scratch_image_manager_;
  MemoryType mem_type_;
};

}  // namespace mace
#endif  // MACE_CORE_RUNTIME_OPENCL_GPU_RUNTIME_H_
