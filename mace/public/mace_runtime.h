//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

// This file defines runtime tuning APIs.
// These APIs are not stable.

#ifndef MACE_PUBLIC_MACE_RUNTIME_H_
#define MACE_PUBLIC_MACE_RUNTIME_H_

namespace mace {

enum GPUPerfHint {
  PERF_DEFAULT = 0,
  PERF_LOW = 1,
  PERF_NORMAL = 2,
  PERF_HIGH = 3
};

enum GPUPriorityHint {
  PRIORITY_DEFAULT = 0,
  PRIORITY_LOW = 1,
  PRIORITY_NORMAL = 2,
  PRIORITY_HIGH = 3
};

enum CPUPowerOption { DEFAULT = 0, HIGH_PERFORMANCE = 1, BATTERY_SAVE = 2 };

void ConfigOpenCLRuntime(GPUPerfHint, GPUPriorityHint);
void ConfigOmpThreadsAndAffinity(int omp_num_threads,
                                 CPUPowerOption power_option);

}  // namespace mace

#endif  // MACE_PUBLIC_MACE_RUNTIME_H_
