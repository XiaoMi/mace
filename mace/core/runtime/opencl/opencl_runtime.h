//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
#define MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_

#include <map>
#include <mutex>

#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_wrapper.h"

namespace mace {

class OpenCLRuntime {
 public:
  static OpenCLRuntime *Get();

  uint32_t GetDeviceMaxWorkGroupSize();
  uint32_t GetKernelMaxWorkGroupSize(const cl::Kernel& kernel);
  cl::Context &context();
  cl::Device &device();
  cl::CommandQueue &command_queue();
  cl::Program &program();
 private:
  OpenCLRuntime(cl::Context context,
                cl::Device device,
                cl::CommandQueue command_queue);
  ~OpenCLRuntime();
  OpenCLRuntime(const OpenCLRuntime&) = delete;
  OpenCLRuntime &operator=(const OpenCLRuntime&) = delete;

 private:
  cl::Context context_;
  cl::Device device_;
  cl::CommandQueue command_queue_;
  cl::Program program_;
  std::once_flag build_flag_;
};

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
