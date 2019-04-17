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

#include "mace/core/device.h"

#include "mace/core/buffer.h"
#include "mace/utils/memory.h"

namespace mace {

CPUDevice::CPUDevice(const int num_threads,
                     const CPUAffinityPolicy policy,
                     utils::ThreadPool *thread_pool)
    : cpu_runtime_(make_unique<CPURuntime>(num_threads,
                                           policy,
                                           thread_pool)),
      scratch_buffer_(make_unique<ScratchBuffer>(GetCPUAllocator())) {}

CPUDevice::~CPUDevice() = default;

CPURuntime *CPUDevice::cpu_runtime() {
  return cpu_runtime_.get();
}

#ifdef MACE_ENABLE_OPENCL
GPURuntime *CPUDevice::gpu_runtime() {
  LOG(FATAL) << "CPU device should not call GPU Runtime";
  return nullptr;
}
#endif

Allocator *CPUDevice::allocator() {
  return GetCPUAllocator();
}

DeviceType CPUDevice::device_type() const {
  return DeviceType::CPU;
}

ScratchBuffer *CPUDevice::scratch_buffer() {
  return scratch_buffer_.get();
}

}  // namespace mace
