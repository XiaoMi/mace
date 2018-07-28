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

#include <string>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/macros.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/types.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

const float kMaxKernelExeTime = 1000.0;  // microseconds

const int32_t kBaseGPUMemCacheSize = 16384;

enum BufferType {
  CONV2D_FILTER = 0,
  IN_OUT_CHANNEL = 1,
  ARGUMENT = 2,
  IN_OUT_HEIGHT = 3,
  IN_OUT_WIDTH = 4,
  WINOGRAD_FILTER = 5,
  DW_CONV2D_FILTER = 6,
  WEIGHT_HEIGHT = 7,
  WEIGHT_WIDTH = 8,
};

void CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                     const BufferType type,
                     std::vector<size_t> *image_shape,
                     const int wino_blk_size = 2);

std::vector<index_t> FormatBufferShape(
    const std::vector<index_t> &buffer_shape,
    const BufferType type);

std::vector<index_t> CalWinogradShape(const std::vector<index_t> &shape,
                                      const BufferType type,
                                      const int wino_blk_size = 2);

std::string DtToCLCMDDt(const DataType dt);

std::string DtToUpstreamCLCMDDt(const DataType dt);

std::string DtToCLDt(const DataType dt);

std::string DtToUpstreamCLDt(const DataType dt);

MaceStatus TuningOrRun3DKernel(const cl::Kernel &kernel,
                               const std::string tuning_key,
                               const uint32_t *gws,
                               const std::vector<uint32_t> &lws,
                               StatsFuture *future);

MaceStatus TuningOrRun2DKernel(const cl::Kernel &kernel,
                               const std::string tuning_key,
                               const uint32_t *gws,
                               const std::vector<uint32_t> &lws,
                               StatsFuture *future);

inline void SetFuture(StatsFuture *future, const cl::Event &event) {
  if (future != nullptr) {
    future->wait_fn = [event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        OpenCLRuntime::Global()->GetCallStats(event, stats);
      }
    };
  }
}

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
