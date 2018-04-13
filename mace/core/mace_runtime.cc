//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/public/mace_runtime.h"
#include "mace/core/runtime/cpu/cpu_runtime.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {

std::shared_ptr<KVStorageFactory> kStorageFactory = nullptr;

void SetGPUHints(GPUPerfHint gpu_perf_hint, GPUPriorityHint gpu_priority_hint) {
  VLOG(1) << "Set GPU configurations, gpu_perf_hint: " << gpu_perf_hint
          << ", gpu_priority_hint: " << gpu_priority_hint;
  OpenCLRuntime::Configure(gpu_perf_hint, gpu_priority_hint);
}

void SetKVStorageFactory(std::shared_ptr<KVStorageFactory> storage_factory) {
  VLOG(1) << "Set internal KV Storage Engine";
  kStorageFactory = storage_factory;
}

MaceStatus SetOpenMPThreadPolicy(int num_threads_hint,
                                 CPUAffinityPolicy policy) {
  VLOG(1) << "Set CPU openmp num_threads_hint: " << num_threads_hint
          << ", affinity policy: " << policy;
  return SetOpenMPThreadsAndAffinityPolicy(num_threads_hint, policy);
}

void SetOpenMPThreadAffinity(int num_threads, const std::vector<int> &cpu_ids) {
  return SetOpenMPThreadsAndAffinityCPUs(num_threads, cpu_ids);
}

MaceStatus GetBigLittleCoreIDs(std::vector<int> *big_core_ids,
                               std::vector<int> *little_core_ids) {
  return GetCPUBigLittleCoreIDs(big_core_ids, little_core_ids);
}

};  // namespace mace
