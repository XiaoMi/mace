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

#include "mace/runtimes/opencl/qc_ion/opencl_qc_ion_executor.h"

#include <string>

#include "mace/runtimes/opencl/core/opencl_extension.h"

namespace mace {

namespace {

uint32_t ParseQcomHostCachePolicy(const std::string &device_extensions) {
  constexpr const char *kQualcommIocoherentStr =
      "cl_qcom_ext_host_ptr_iocoherent";

  auto pos = device_extensions.find(kQualcommIocoherentStr);
  if (false && pos != std::string::npos) {
    // Will lead to computing mistake on some Qualcomm platform.
    return CL_MEM_HOST_IOCOHERENT_QCOM;
  } else {
    return CL_MEM_HOST_WRITEBACK_QCOM;
  }
}

std::string QcomHostCachePolicyToString(uint32_t policy) {
  switch (policy) {
    case CL_MEM_HOST_IOCOHERENT_QCOM: return "CL_MEM_HOST_IOCOHERENT_QCOM";
    case CL_MEM_HOST_WRITEBACK_QCOM: return "CL_MEM_HOST_WRITEBACK_QCOM";
    default: return MakeString("UNKNOWN: ", policy);
  }
}
}  // namespace

OpenclQcIonExecutor::OpenclQcIonExecutor() : OpenclExecutor() {}

OpenclQcIonExecutor *OpenclQcIonExecutor::Get(OpenclExecutor *executor) {
  return static_cast<OpenclQcIonExecutor *>(executor);
}

IONType OpenclQcIonExecutor::ion_type() const {
  return IONType::QUALCOMM_ION;
}

uint32_t OpenclQcIonExecutor::qcom_ext_mem_padding() const {
  return qcom_ext_mem_padding_;
}

uint32_t OpenclQcIonExecutor::qcom_page_size() const {
  return qcom_page_size_;
}

uint32_t OpenclQcIonExecutor::qcom_host_cache_policy() const {
  return qcom_host_cache_policy_;
}

void OpenclQcIonExecutor::InitGpuDeviceProperty(const cl::Device &device) {
  OpenclExecutor::InitGpuDeviceProperty(device);

  const auto device_extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
  qcom_ext_mem_padding_ = 0;
  cl_int err = device.getInfo(CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM,
                              &qcom_ext_mem_padding_);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to get CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM "
               << OpenCLErrorToString(err);
  }

  qcom_page_size_ = 4096;
  err = device.getInfo(CL_DEVICE_PAGE_SIZE_QCOM, &qcom_page_size_);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to get CL_DEVICE_PAGE_SIZE_QCOM: "
               << OpenCLErrorToString(err);
  }

  qcom_host_cache_policy_ = ParseQcomHostCachePolicy(device_extensions);

  VLOG(1) << "Using QUALCOMM ION buffer with padding size: "
          << qcom_ext_mem_padding_ << ", page size: " << qcom_page_size_
          << ", with host cache policy: "
          << QcomHostCachePolicyToString(qcom_host_cache_policy_);
}

}  // namespace mace
