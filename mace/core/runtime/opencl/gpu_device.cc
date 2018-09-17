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

#include "mace/core/runtime/opencl/gpu_device.h"

namespace mace {

GPUDevice::GPUDevice(Tuner<uint32_t> *tuner,
                     KVStorage *opencl_cache_storage,
                     const GPUPriorityHint priority,
                     const GPUPerfHint perf,
                     KVStorage *opencl_binary_storage,
                     const int num_threads,
                     CPUAffinityPolicy cpu_affinity_policy,
                     bool use_gemmlowp) :
    CPUDevice(num_threads, cpu_affinity_policy, use_gemmlowp),
    runtime_(new OpenCLRuntime(opencl_cache_storage, priority, perf,
                               opencl_binary_storage, tuner)),
    allocator_(new OpenCLAllocator(runtime_.get())) {}

GPUDevice::~GPUDevice() = default;

OpenCLRuntime* GPUDevice::opencl_runtime() {
  return runtime_.get();
}

Allocator* GPUDevice::allocator() {
  return allocator_.get();
}

DeviceType GPUDevice::device_type() const {
  return DeviceType::GPU;
}

}  // namespace mace
