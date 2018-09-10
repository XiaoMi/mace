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

#include "mace/core/device_context.h"

#include <sys/stat.h>

namespace mace {

namespace {

const char *kPrecompiledProgramFileName = "mace_cl_compiled_program.bin";

std::string FindFirstExistPath(const std::vector<std::string> &paths) {
  std::string result;
  struct stat st;
  for (auto path : paths) {
    if (stat(path.c_str(), &st) == 0) {
      if (S_ISREG(st.st_mode)) {
        result = path;
        break;
      }
    }
  }
  return result;
}
}  // namespace

GPUContext::GPUContext(const std::string &storage_path,
                       const std::vector<std::string> &opencl_binary_paths,
                       const std::string &opencl_parameter_path)
    : storage_factory_(new FileStorageFactory(storage_path)),
      opencl_tuner_(new Tuner<uint32_t>(opencl_parameter_path)) {

  if (!storage_path.empty()) {
    opencl_cache_storage_ =
        storage_factory_->CreateStorage(kPrecompiledProgramFileName);
  }

  std::string precompiled_binary_path =
      FindFirstExistPath(opencl_binary_paths);
  if (!precompiled_binary_path.empty()) {
    opencl_binary_storage_.reset(
        new FileStorage(precompiled_binary_path));
  }
}

GPUContext::~GPUContext() = default;

KVStorage *GPUContext::opencl_binary_storage() {
  return opencl_binary_storage_.get();
}

KVStorage *GPUContext::opencl_cache_storage() {
  return opencl_cache_storage_.get();
}

Tuner<uint32_t> *GPUContext::opencl_tuner() {
  return opencl_tuner_.get();
}

}  // namespace mace
