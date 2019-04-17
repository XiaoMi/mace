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

#ifndef MACE_CORE_DEVICE_H_
#define MACE_CORE_DEVICE_H_

#include <memory>

#include "mace/core/runtime/cpu/cpu_runtime.h"
#include "mace/core/allocator.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/gpu_runtime.h"
#endif

namespace mace {

class ScratchBuffer;

class Device {
 public:
  virtual ~Device() {}

#ifdef MACE_ENABLE_OPENCL
  virtual GPURuntime *gpu_runtime() = 0;
#endif  // MACE_ENABLE_OPENCL
  virtual CPURuntime *cpu_runtime() = 0;

  virtual Allocator *allocator() = 0;
  virtual DeviceType device_type() const = 0;
  virtual ScratchBuffer *scratch_buffer() = 0;
};

class CPUDevice : public Device {
 public:
  CPUDevice(const int num_threads,
            const CPUAffinityPolicy policy,
            utils::ThreadPool *thread_pool);
  virtual ~CPUDevice();

#ifdef MACE_ENABLE_OPENCL
  GPURuntime *gpu_runtime() override;
#endif
  CPURuntime *cpu_runtime() override;

  Allocator *allocator() override;
  DeviceType device_type() const override;
  ScratchBuffer *scratch_buffer() override;

 private:
  std::unique_ptr<CPURuntime> cpu_runtime_;
  std::unique_ptr<ScratchBuffer> scratch_buffer_;
};

}  // namespace mace
#endif  // MACE_CORE_DEVICE_H_
