//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_OPENCL_HELPER_H_
#define MACE_KERNELS_OPENCL_HELPER_H_

#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/types.h"
#include "mace/utils/utils.h"
#include "mace/core/future.h"

namespace mace {
namespace kernels {

const float kMaxKernelExeTime = 1000.0; // microseconds

enum BufferType {
  FILTER = 0,
  IN_OUT= 1,
  ARGUMENT = 2
};

void CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                     const BufferType type,
                     std::vector<size_t> &image_shape);

std::string DtToCLCMDDt(const DataType dt);

std::string DtToUpstreamCLCMDDt(const DataType dt);

std::string DtToCLDt(const DataType dt);

std::string DtToUpstreamCLDt(const DataType dt);

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

namespace {
template<typename T>
void AppendToStream(std::stringstream *ss, const std::string &delimiter, T v) {
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
}  // namespace

template<typename... Args>
std::string Concat(Args... args) {
  std::stringstream ss;
  AppendToStream(&ss, "_", args...);
  return ss.str();
}

}  // namespace kernels
} //  namespace mace
#endif //  MACE_KERNELS_OPENCL_HELPER_H_
