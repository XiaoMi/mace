//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

// This file defines runtime tuning APIs.
// These APIs are not stable.

#ifndef MACE_PUBLIC_MACE_RUNTIME_H_
#define MACE_PUBLIC_MACE_RUNTIME_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

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

class KVStorageEngine {
 public:
  virtual void Write(
      const std::map<std::string, std::vector<unsigned char>> &data) = 0;
  virtual void Read(
      std::map<std::string, std::vector<unsigned char>> *data) = 0;
};

void ConfigOpenCLRuntime(GPUPerfHint, GPUPriorityHint);
void ConfigKVStorageEngine(std::shared_ptr<KVStorageEngine> storage_engine);
void ConfigOmpThreads(int omp_num_threads);
void ConfigCPUPowerOption(CPUPowerOption power_option);


}  // namespace mace

#endif  // MACE_PUBLIC_MACE_RUNTIME_H_
