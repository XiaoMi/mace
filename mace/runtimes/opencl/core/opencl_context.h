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

#ifndef MACE_RUNTIMES_OPENCL_CORE_OPENCL_CONTEXT_H_
#define MACE_RUNTIMES_OPENCL_CORE_OPENCL_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mace/core/kv_storage.h"
#include "mace/utils/tuner.h"

namespace mace {

class OpenclContext {
 public:
  OpenclContext(const std::string &storage_path = "",
                const std::string &opencl_cache_full_path = "",
                const OpenCLCacheReusePolicy &opencl_cache_reuse_policy =
                    OpenCLCacheReusePolicy::REUSE_SAME_GPU,
                const std::vector<std::string> &opencl_binary_path = {},
                const std::string &opencl_parameter_path = "",
                const unsigned char *opencl_binary_ptr = nullptr,
                const size_t opencl_binary_size = 0,
                const unsigned char *opencl_parameter_ptr = nullptr,
                const size_t opencl_parameter_size = 0);
  ~OpenclContext();

  std::shared_ptr<KVStorage> opencl_binary_storage();
  std::shared_ptr<KVStorage> opencl_cache_storage();
  std::shared_ptr<Tuner<uint32_t>> opencl_tuner();
  OpenCLCacheReusePolicy opencl_cache_reuse_policy();

 private:
  std::unique_ptr<KVStorageFactory> storage_factory_;
  std::shared_ptr<Tuner<uint32_t>> opencl_tuner_;
  std::shared_ptr<KVStorage> opencl_binary_storage_;
  std::shared_ptr<KVStorage> opencl_cache_storage_;
  OpenCLCacheReusePolicy opencl_cache_reuse_policy_;
};

}  // namespace mace
#endif  // MACE_RUNTIMES_OPENCL_CORE_OPENCL_CONTEXT_H_
