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

#include "mace/core/runtime/opencl/gpu_runtime.h"

#include "mace/core/runtime/opencl/scratch_image.h"

namespace mace {

GPURuntime::GPURuntime(mace::OpenCLRuntime *runtime)
    : runtime_(runtime),
      scratch_image_manager_(new ScratchImageManager),
      mem_type_(MemoryType::GPU_IMAGE) {}

GPURuntime::~GPURuntime() = default;

OpenCLRuntime* GPURuntime::opencl_runtime() {
  return runtime_;
}

ScratchImageManager* GPURuntime::scratch_image_manager() const {
  return scratch_image_manager_.get();
}

bool GPURuntime::UseImageMemory() {
  return this->mem_type_ == MemoryType::GPU_IMAGE;
}

void GPURuntime::set_mem_type(MemoryType type) {
  this->mem_type_ = type;
}


}  // namespace mace
