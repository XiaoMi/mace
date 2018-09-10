// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_CORE_RUNTIME_OPENCL_GPU_DEVICE_H_
#define MACE_CORE_RUNTIME_OPENCL_GPU_DEVICE_H_

#include <memory>

#include "mace/core/device_context.h"
#include "mace/core/device.h"
#include "mace/core/runtime/opencl/opencl_allocator.h"

namespace mace {

class GPUDevice : public CPUDevice {
 public:
  GPUDevice(Tuner<uint32_t> *tuner,
            KVStorage *opencl_cache_storage = nullptr,
            const GPUPriorityHint priority = GPUPriorityHint::PRIORITY_LOW,
            const GPUPerfHint perf = GPUPerfHint::PERF_NORMAL,
            KVStorage *opencl_binary_storage = nullptr,
            const int num_threads = -1);
  ~GPUDevice();
  OpenCLRuntime *opencl_runtime() override;
  Allocator *allocator() override;
  DeviceType device_type() const override;
 private:
  std::unique_ptr<OpenCLRuntime> runtime_;
  std::unique_ptr<OpenCLAllocator> allocator_;
};

}  // namespace mace
#endif  // MACE_CORE_RUNTIME_OPENCL_GPU_DEVICE_H_
