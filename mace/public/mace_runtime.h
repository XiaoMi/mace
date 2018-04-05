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

class KVStorage {
 public:
  virtual void Load() = 0;
  virtual bool Insert(const std::string &key,
                      const std::vector<unsigned char> &value) = 0;
  virtual std::vector<unsigned char> *Find(const std::string &key) = 0;
  virtual void Flush() = 0;
};

class KVStorageFactory {
 public:
  virtual std::unique_ptr<KVStorage> CreateStorage(const std::string &name) = 0;
};

class FileStorageFactory : public KVStorageFactory {
 public:
  explicit FileStorageFactory(const std::string &path);

  ~FileStorageFactory();

  std::unique_ptr<KVStorage> CreateStorage(const std::string &name) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

void ConfigKVStorageFactory(std::shared_ptr<KVStorageFactory> storage_factory);

void ConfigOpenCLRuntime(GPUPerfHint, GPUPriorityHint);
void ConfigOmpThreads(int omp_num_threads);
void ConfigCPUPowerOption(CPUPowerOption power_option);


}  // namespace mace

#endif  // MACE_PUBLIC_MACE_RUNTIME_H_
