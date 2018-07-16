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

#include "mace/core/macros.h"
#include "mace/core/file_storage.h"
#include "mace/core/runtime/cpu/cpu_runtime.h"
#include "mace/public/mace_runtime.h"
#include "mace/utils/logging.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/opencl_runtime.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {

class FileStorageFactory::Impl {
 public:
  explicit Impl(const std::string &path);

  std::unique_ptr<KVStorage> CreateStorage(const std::string &name);

 private:
  std::string path_;
};

FileStorageFactory::Impl::Impl(const std::string &path): path_(path) {}

std::unique_ptr<KVStorage> FileStorageFactory::Impl::CreateStorage(
    const std::string &name) {
  return std::move(std::unique_ptr<KVStorage>(
      new FileStorage(path_ + "/" + name)));
}

FileStorageFactory::FileStorageFactory(const std::string &path):
    impl_(new FileStorageFactory::Impl(path)) {}

FileStorageFactory::~FileStorageFactory() = default;

std::unique_ptr<KVStorage> FileStorageFactory::CreateStorage(
    const std::string &name) {
  return impl_->CreateStorage(name);
}

extern std::shared_ptr<KVStorageFactory> kStorageFactory;

void SetKVStorageFactory(std::shared_ptr<KVStorageFactory> storage_factory) {
  VLOG(1) << "Set internal KV Storage Engine";
  kStorageFactory = storage_factory;
}

// Set OpenCL Compiled Binary paths, just call once. (Not thread-safe)
void SetOpenCLBinaryPaths(const std::vector<std::string> &paths) {
#ifdef MACE_ENABLE_OPENCL
  OpenCLRuntime::ConfigureOpenCLBinaryPath(paths);
#else
  MACE_UNUSED(paths);
#endif  // MACE_ENABLE_OPENCL
}

extern std::string kOpenCLParameterPath;

void SetOpenCLParameterPath(const std::string &path) {
#ifdef MACE_ENABLE_OPENCL
  kOpenCLParameterPath = path;
#else
  MACE_UNUSED(path);
#endif  // MACE_ENABLE_OPENCL
}

void SetGPUHints(GPUPerfHint gpu_perf_hint, GPUPriorityHint gpu_priority_hint) {
#ifdef MACE_ENABLE_OPENCL
  VLOG(1) << "Set GPU configurations, gpu_perf_hint: " << gpu_perf_hint
          << ", gpu_priority_hint: " << gpu_priority_hint;
  OpenCLRuntime::Configure(gpu_perf_hint, gpu_priority_hint);
#else
  MACE_UNUSED(gpu_perf_hint);
  MACE_UNUSED(gpu_priority_hint);
#endif  // MACE_ENABLE_OPENCL
}

MaceStatus SetOpenMPThreadPolicy(int num_threads_hint,
                                 CPUAffinityPolicy policy) {
  VLOG(1) << "Set OpenMP threads number hint: " << num_threads_hint
          << ", affinity policy: " << policy;
  return SetOpenMPThreadsAndAffinityPolicy(num_threads_hint, policy);
}

MaceStatus SetOpenMPThreadAffinity(int num_threads,
                                   const std::vector<int> &cpu_ids) {
  return SetOpenMPThreadsAndAffinityCPUs(num_threads, cpu_ids);
}

MaceStatus GetBigLittleCoreIDs(std::vector<int> *big_core_ids,
                               std::vector<int> *little_core_ids) {
  return GetCPUBigLittleCoreIDs(big_core_ids, little_core_ids);
}


};  // namespace mace
