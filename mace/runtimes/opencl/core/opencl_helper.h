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

#ifndef MACE_RUNTIMES_OPENCL_CORE_OPENCL_HELPER_H_
#define MACE_RUNTIMES_OPENCL_CORE_OPENCL_HELPER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/memory/buffer.h"
#include "mace/core/ops/op_context.h"
#include "mace/core/types.h"
#include "mace/runtimes/opencl/core/cl2_header.h"
#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/core/opencl_util.h"
#include "mace/utils/macros.h"
#include "mace/utils/memory.h"
#include "mace/utils/math.h"

namespace mace {

class OpContext;
class OpenclRuntime;

MaceStatus InitOorcFlag(OpenclRuntime *opencl_runtime, Buffer *buffer,
                        cl::Kernel *kernel);
MaceStatus InitOorcFlag(OpContext *context, Buffer *buffer, cl::Kernel *kernel);
MaceStatus ValidateOorcFlag(OpenclRuntime *opencl_runtime, Buffer *buffer);
MaceStatus ValidateOorcFlag(OpContext *context, Buffer *buffer);

// oorc for 'Out Of Range Check'
#define MACE_OUT_OF_RANGE_DEFINITION           \
  Buffer oorc_flag(GPU_BUFFER, DT_INT32, {1});

#define MACE_OUT_OF_RANGE_CONFIG                   \
  if (executor->IsOutOfRangeCheckEnabled()) {      \
    built_options.emplace("-DOUT_OF_RANGE_CHECK"); \
  }

// @Deprecated, should not use context,
#define MACE_OUT_OF_RANGE_INIT(kernel) \
  MACE_CHECK_SUCCESS(InitOorcFlag(context, &oorc_flag, (&kernel)))

#define MACE_OUT_OF_RANGE_INITION(kernel) \
  MACE_CHECK_SUCCESS(InitOorcFlag(opencl_runtime, &oorc_flag, (&kernel)))

#define MACE_OUT_OF_RANGE_SET_ARGS(kernel)             \
  if (executor->IsOutOfRangeCheckEnabled()) {          \
    (kernel).setArg(idx++,                             \
    *(oorc_flag.mutable_memory<cl::Buffer>()));       \
  }

#define MACE_BUFF_OUT_OF_RANGE_SET_ARGS(kernel, size)     \
  if (executor->IsOutOfRangeCheckEnabled()) {             \
    (kernel).setArg(idx++,                                \
    *(oorc_flag.mutable_memory<cl::Buffer>()));          \
    (kernel).setArg(idx++, static_cast<int>(size));       \
  }

// @Deprecated, should not use context,
#define MACE_OUT_OF_RANGE_VALIDATION \
  ValidateOorcFlag(context, &oorc_flag)

#define MACE_OUT_OF_RANGE_VALIDATE \
  ValidateOorcFlag(opencl_runtime, &oorc_flag)

#define MACE_NON_UNIFORM_WG_CONFIG                      \
  if (executor->IsNonUniformWorkgroupsSupported()) {    \
    built_options.emplace("-DNON_UNIFORM_WORK_GROUP");  \
  }

#define MACE_SET_3D_GWS_ARGS(kernel, gws) \
  (kernel).setArg(idx++, (gws)[0]);       \
  (kernel).setArg(idx++, (gws)[1]);       \
  (kernel).setArg(idx++, (gws)[2]);

#define MACE_SET_2D_GWS_ARGS(kernel, gws) \
  (kernel).setArg(idx++, (gws)[0]);       \
  (kernel).setArg(idx++, (gws)[1]);

// Max execution time of OpenCL kernel for tuning to prevent UI stuck.
const float kMaxKernelExecTime = 1000.0;  // microseconds

// Base GPU cache size used for computing local work group size.
const int32_t kBaseGPUMemCacheSize = 16384;

std::vector<index_t> FormatBufferShape(
    const std::vector<index_t> &buffer_shape,
    const BufferContentType type);

// CPU data type to OpenCL command data type
std::string DtToCLCMDDt(const DataType dt);

// CPU data type to OpenCL data type
std::string DtToCLDt(const DataType dt);

// CPU data type to OpenCL condition data type used in select
// e.g. half -> float
std::string DtToCLCondDt(const DataType dt);

// Tuning or Run OpenCL kernel with 3D work group size
MaceStatus TuningOrRun3DKernel(OpenclExecutor *executor,
                               const cl::Kernel &kernel,
                               const std::string tuning_key,
                               const uint32_t *gws,
                               const std::vector<uint32_t> &lws,
                               StatsFuture *future,
                               OpContext *context = nullptr);

// Tuning or Run OpenCL kernel with 2D work group size
MaceStatus TuningOrRun2DKernel(OpenclExecutor *executor,
                               const cl::Kernel &kernel,
                               const std::string tuning_key,
                               const uint32_t *gws,
                               const std::vector<uint32_t> &lws,
                               StatsFuture *future,
                               OpContext *context = nullptr);

// Check whether limit OpenCL kernel time flag open.
inline bool LimitKernelTime() {
  const char *flag = getenv("MACE_LIMIT_OPENCL_KERNEL_TIME");
  return flag != nullptr && strlen(flag) == 1 && flag[0] == '1';
}

template<typename T>
bool IsVecEqual(const std::vector<T> &input0, const std::vector<T> &input1) {
  return ((input0.size() == input1.size()) &&
      (std::equal(input0.begin(), input0.end(), input1.begin())));
}

template <typename T>
bool IsResetArgsNeeded(OpContext *context,
                       const std::vector<T> &input0,
                       const std::vector<T> &input1) {
  return !context->runtime()->IntermediateBufferStable(context) ||
      !IsVecEqual(input0, input1);
}

template <typename T>
void AppendToStream(std::stringstream *ss, const std::string &delimiter, T v) {
  MACE_UNUSED(delimiter);
  (*ss) << v;
}

template<typename T, typename... Args>
void AppendToStream(std::stringstream *ss,
                    const std::string &delimiter,
                    T first,
                    Args... args) {
  (*ss) << first << delimiter;
  AppendToStream(ss, delimiter, args...);
}

template<typename... Args>
std::string Concat(Args... args) {
  std::stringstream ss;
  AppendToStream(&ss, "_", args...);
  return ss.str();
}

std::vector<uint32_t> Default3DLocalWS(OpenclExecutor *executor,
                                       const uint32_t *gws,
                                       const uint32_t kwg_size);

}  // namespace mace
#endif  // MACE_RUNTIMES_OPENCL_CORE_OPENCL_HELPER_H_
