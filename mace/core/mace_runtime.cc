//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/public/mace_runtime.h"
#include "mace/core/runtime/cpu/cpu_runtime.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {

void ConfigOpenCLRuntime(GPUPerfHint gpu_perf_hint,
                         GPUPriorityHint gpu_priority_hint) {
  VLOG(1) << "Set GPU configurations, gpu_perf_hint: " << gpu_perf_hint
          << ", gpu_priority_hint: " << gpu_priority_hint;
  OpenCLRuntime::Configure(gpu_perf_hint, gpu_priority_hint);
}

void ConfigOmpThreadsAndAffinity(int omp_num_threads,
                                 CPUPowerOption power_option) {
  VLOG(1) << "Config CPU Runtime: omp_num_threads: " << omp_num_threads
          << ", cpu_power_option: " << power_option;
  SetOmpThreadsAndAffinity(omp_num_threads, power_option);
}

};  // namespace mace
