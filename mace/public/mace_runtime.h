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

#include "mace/public/mace.h"

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

enum CPUAffinityPolicy {
  AFFINITY_DEFAULT = 0,
  AFFINITY_BIG_ONLY = 1,
  AFFINITY_LITTLE_ONLY = 2,
};

class KVStorage {
 public:
  // return: 0 for success, -1 for error
  virtual int Load() = 0;
  virtual bool Insert(const std::string &key,
                      const std::vector<unsigned char> &value) = 0;
  virtual const std::vector<unsigned char> *Find(const std::string &key) = 0;
  // return: 0 for success, -1 for error
  virtual int Flush() = 0;
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

// Set KV store factory used as OpenCL cache
void SetKVStorageFactory(std::shared_ptr<KVStorageFactory> storage_factory);

// Set GPU hints, currently only supports Adreno GPU
void SetGPUHints(GPUPerfHint perf_hint, GPUPriorityHint priority_hint);

// Set OpenMP threads number and affinity policy.
//
// num_threads_hint is only a hint, the function can change it when it's larger
// than 0. When num_threads_hint is not positive, the function will set the
// threads number equaling to the number of big + little, big or little cores
// according to the policy.
//
// This function may not work well on some ships (e.g. MTK), and in such
// cases (when it returns error MACE_INVALID_ARGS) you may try to use
// SetOpenMPThreadAffinity to set affinity manually, or just set default policy.
MaceStatus SetOpenMPThreadPolicy(int num_threads_hint,
                                 CPUAffinityPolicy policy);

// Set OpenMP threads number and processor affinity
// This function may not work well on some chips (e.g. MTK). Set thread affinity
// to offline cores may fail or run unexpectedly. In such cases, please use
// SetOpenMPThreadPolicy with default policy instead.
void SetOpenMPThreadAffinity(int num_threads, const std::vector<int> &cpu_ids);

// Get ARM big.LITTLE configuration.
//
// This function may not work well on some chips (e.g. MTK) and miss the
// offline cores, and the user should detect the configurations manually
// in such case(when it returns error MACE_INVALID_ARGS).
//
// If all cpu's frequencies are equal(i.e. all cores are the same),
// big_core_ids and little_core_ids will be set to all cpu ids.
MaceStatus GetBigLittleCoreIDs(std::vector<int> *big_core_ids,
                               std::vector<int> *little_core_ids);

}  // namespace mace

#endif  // MACE_PUBLIC_MACE_RUNTIME_H_
