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

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

OpTestContext *OpTestContext::Get(int num_threads,
                                  CPUAffinityPolicy cpu_affinity_policy,
                                  bool use_gemmlowp) {
  static OpTestContext instance(num_threads,
                                cpu_affinity_policy,
                                use_gemmlowp);
  return &instance;
}

std::shared_ptr<GPUContext> OpTestContext::gpu_context() const {
  return gpu_context_;
}

Device *OpTestContext::GetDevice(DeviceType device_type) {
  return device_map_[device_type].get();
}

OpTestContext::OpTestContext(int num_threads,
                             CPUAffinityPolicy cpu_affinity_policy,
                             bool use_gemmlowp)
    : gpu_context_(new GPUContext()) {
  device_map_[DeviceType::CPU] = std::unique_ptr<Device>(
      new CPUDevice(num_threads,
                    cpu_affinity_policy,
                    use_gemmlowp));

  device_map_[DeviceType::GPU] = std::unique_ptr<Device>(
      new GPUDevice(gpu_context_->opencl_tuner(),
                    gpu_context_->opencl_cache_storage(),
                    GPUPriorityHint::PRIORITY_NORMAL));
}

}  // namespace test
}  // namespace ops
}  // namespace mace
