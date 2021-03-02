// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_RUNTIMES_OPENCL_QC_ION_OPENCL_QC_ION_EXECUTOR_H_
#define MACE_RUNTIMES_OPENCL_QC_ION_OPENCL_QC_ION_EXECUTOR_H_

#include <memory>

#include "mace/runtimes/opencl/core/opencl_executor.h"

namespace mace {

class OpenclQcIonExecutor : public OpenclExecutor {
 public:
  explicit OpenclQcIonExecutor(
      std::shared_ptr<KVStorage> cache_storage = nullptr,
      std::shared_ptr<KVStorage> precompiled_binary_storage = nullptr,
      std::shared_ptr<Tuner<uint32_t>> tuner = nullptr);
  virtual ~OpenclQcIonExecutor() = default;

  static OpenclQcIonExecutor *Get(OpenclExecutor *executor);

  IONType ion_type() const override;

  uint32_t qcom_ext_mem_padding() const;
  uint32_t qcom_page_size() const;
  uint32_t qcom_host_cache_policy() const;

 protected:
  void InitGpuDeviceProperty(const cl::Device &device) override;

 private:
  uint32_t qcom_ext_mem_padding_;
  uint32_t qcom_page_size_;
  uint32_t qcom_host_cache_policy_;
};

}  // namespace mace

#endif  // MACE_RUNTIMES_OPENCL_QC_ION_OPENCL_QC_ION_EXECUTOR_H_
