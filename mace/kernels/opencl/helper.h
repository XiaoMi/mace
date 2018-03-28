//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_OPENCL_HELPER_H_
#define MACE_KERNELS_OPENCL_HELPER_H_

#include <string>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/types.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

const float kMaxKernelExeTime = 1000.0;  // microseconds

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
                     std::vector<size_t> *image_shape);

std::vector<index_t> CalWinogradShape(const std::vector<index_t> &shape,
                                      const BufferType type);

std::string DtToCLCMDDt(const DataType dt);

std::string DtToUpstreamCLCMDDt(const DataType dt);

std::string DtToCLDt(const DataType dt);

std::string DtToUpstreamCLDt(const DataType dt);

void TuningOrRun3DKernel(const cl::Kernel &kernel,
                         const std::string tuning_key,
                         const uint32_t *gws,
                         const std::vector<uint32_t> &lws,
                         StatsFuture *future);

void TuningOrRun2DKernel(const cl::Kernel &kernel,
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
bool IsVecEqual(const std::vector<T> &input0,
                const std::vector<T> &input1) {
  return ((input0.size() == input1.size()) &&
      (std::equal(input0.begin(), input0.end(), input1.begin())));
}

template <typename T>
void AppendToStream(std::stringstream *ss, const std::string &delimiter, T v) {
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

const bool IsQualcommOpenCL200();

}  // namespace kernels
}  // namespace mace
#endif  // MACE_KERNELS_OPENCL_HELPER_H_
