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

#ifndef MACE_KERNELS_OPENCL_HELPER_H_
#define MACE_KERNELS_OPENCL_HELPER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/macros.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/types.h"
#include "mace/kernels/opencl/common.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

#define OUT_OF_RANGE_CONFIG(kernel_error)                   \
  if (runtime->IsOutOfRangeCheckEnabled()) {                \
    built_options.emplace("-DOUT_OF_RANGE_CHECK");          \
    (kernel_error) = std::move(std::unique_ptr<Buffer>(     \
        new Buffer(GetDeviceAllocator(DeviceType::GPU))));  \
    MACE_RETURN_IF_ERROR((kernel_error)->Allocate(1));      \
    (kernel_error)->Map(nullptr);                           \
    *((kernel_error)->mutable_data<char>()) = 0;            \
    (kernel_error)->UnMap();                                \
  }

#define OUT_OF_RANGE_SET_ARG                                \
  if (runtime->IsOutOfRangeCheckEnabled()) {                \
    kernel_.setArg(idx++,                                   \
    *(static_cast<cl::Buffer *>(kernel_error_->buffer()))); \
  }

#define OUT_OF_RANGE_SET_ARG_PTR                              \
  if (runtime->IsOutOfRangeCheckEnabled()) {                  \
    kernel->setArg(idx++,                                     \
    *(static_cast<cl::Buffer *>((*kernel_error)->buffer()))); \
  }

#define OUT_OF_RANGE_VALIDATION(kernel_error)                              \
  if (runtime->IsOutOfRangeCheckEnabled()) {                               \
    (kernel_error)->Map(nullptr);                                          \
    char *kerror_code = (kernel_error)->mutable_data<char>();              \
    MACE_CHECK(*kerror_code == 0, "Kernel error code: ", *kerror_code);\
    (kernel_error)->UnMap();                                               \
  }

#define NON_UNIFORM_WG_CONFIG                           \
  if (runtime->IsNonUniformWorkgroupsSupported()) {     \
    built_options.emplace("-DNON_UNIFORM_WORK_GROUP");  \
  }

#define SET_3D_GWS_ARGS(kernel) \
  kernel.setArg(idx++, gws[0]); \
  kernel.setArg(idx++, gws[1]); \
  kernel.setArg(idx++, gws[2]);

#define SET_2D_GWS_ARGS(kernel) \
  kernel.setArg(idx++, gws[0]); \
  kernel.setArg(idx++, gws[1]);

#define SET_3D_GWS_ARGS_PTR(kernel, gws)  \
  kernel->setArg(idx++, (gws)[0]);        \
  kernel->setArg(idx++, (gws)[1]);        \
  kernel->setArg(idx++, (gws)[2]);

#define SET_2D_GWS_ARGS_PTR(kernel, gws)  \
  kernel->setArg(idx++, (gws)[0]);        \
  kernel->setArg(idx++, (gws)[1]);

// Max execution time of OpenCL kernel for tuning to prevent UI stuck.
const float kMaxKernelExecTime = 1000.0;  // microseconds

// Base GPU cache size used for computing local work group size.
const int32_t kBaseGPUMemCacheSize = 16384;

void CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                     const BufferType type,
                     std::vector<size_t> *image_shape,
                     const int wino_blk_size = 2);

std::vector<index_t> FormatBufferShape(
    const std::vector<index_t> &buffer_shape,
    const BufferType type);

// CPU data type to OpenCL command data type
std::string DtToCLCMDDt(const DataType dt);

// CPU data type to upward compatible OpenCL command data type
// e.g. half -> float
std::string DtToUpCompatibleCLCMDDt(const DataType dt);

// CPU data type to OpenCL data type
std::string DtToCLDt(const DataType dt);

// CPU data type to upward compatible OpenCL data type
// e.g. half -> float
std::string DtToUpCompatibleCLDt(const DataType dt);

// Tuning or Run OpenCL kernel with 3D work group size
MaceStatus TuningOrRun3DKernel(const cl::Kernel &kernel,
                               const std::string tuning_key,
                               const uint32_t *gws,
                               const std::vector<uint32_t> &lws,
                               StatsFuture *future);

// Tuning or Run OpenCL kernel with 2D work group size
MaceStatus TuningOrRun2DKernel(const cl::Kernel &kernel,
                               const std::string tuning_key,
                               const uint32_t *gws,
                               const std::vector<uint32_t> &lws,
                               StatsFuture *future);

// Check whether limit OpenCL kernel time flag open.
inline bool LimitKernelTime() {
  const char *flag = getenv("MACE_LIMIT_OPENCL_KERNEL_TIME");
  return flag != nullptr && strlen(flag) == 1 && flag[0] == '1';
}

template <typename T>
bool IsVecEqual(const std::vector<T> &input0, const std::vector<T> &input1) {
  return ((input0.size() == input1.size()) &&
          (std::equal(input0.begin(), input0.end(), input1.begin())));
}

template <typename T>
void AppendToStream(std::stringstream *ss, const std::string &delimiter, T v) {
  MACE_UNUSED(delimiter);
  (*ss) << v;
}

template <typename T, typename... Args>
void AppendToStream(std::stringstream *ss,
                    const std::string &delimiter,
                    T first,
                    Args... args) {
  (*ss) << first << delimiter;
  AppendToStream(ss, delimiter, args...);
}

template <typename... Args>
std::string Concat(Args... args) {
  std::stringstream ss;
  AppendToStream(&ss, "_", args...);
  return ss.str();
}

std::vector<uint32_t> Default3DLocalWS(const uint32_t *gws,
                                       const uint32_t kwg_size);
}  // namespace kernels
}  // namespace mace
#endif  // MACE_KERNELS_OPENCL_HELPER_H_
