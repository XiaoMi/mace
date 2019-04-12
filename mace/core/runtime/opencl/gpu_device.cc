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

#include "mace/core/runtime/opencl/gpu_device.h"

#include "mace/core/buffer.h"

namespace mace {

GPUDevice::GPUDevice(std::shared_ptr<Tuner<uint32_t>> tuner,
                     std::shared_ptr<KVStorage> opencl_cache_storage,
                     const GPUPriorityHint priority,
                     const GPUPerfHint perf,
                     std::shared_ptr<KVStorage> opencl_binary_storage,
                     const int num_threads,
                     CPUAffinityPolicy cpu_affinity_policy,
                     utils::ThreadPool *thread_pool) :
    CPUDevice(num_threads,
              cpu_affinity_policy,
              thread_pool),
    runtime_(new OpenCLRuntime(opencl_cache_storage, priority, perf,
                               opencl_binary_storage, tuner)),
    allocator_(new OpenCLAllocator(runtime_.get())),
    scratch_buffer_(new ScratchBuffer(allocator_.get())),
    gpu_runtime_(new GPURuntime(runtime_.get())) {}

GPUDevice::~GPUDevice() = default;

GPURuntime *GPUDevice::gpu_runtime() {
  return gpu_runtime_.get();
}

Allocator *GPUDevice::allocator() {
  return allocator_.get();
}

DeviceType GPUDevice::device_type() const {
  return DeviceType::GPU;
}

ScratchBuffer *GPUDevice::scratch_buffer() {
  return scratch_buffer_.get();
}

}  // namespace mace
