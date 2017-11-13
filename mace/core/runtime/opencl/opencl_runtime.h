//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
#define MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_

#include <map>
#include <mutex>
#include <unordered_map>

#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_wrapper.h"

namespace mace {

class OpenCLRuntime {
 public:
  static OpenCLRuntime *Get();

  cl::Context &context();
  cl::Device &device();
  cl::CommandQueue &command_queue();
  cl::Program &program();

  uint32_t GetDeviceMaxWorkGroupSize();
  uint32_t GetKernelMaxWorkGroupSize(const cl::Kernel& kernel);
  cl::Kernel BuildKernel(const std::string &kernel_name,
                         const std::set<std::string> &build_options);
 private:
  OpenCLRuntime(cl::Context context,
                cl::Device device,
                cl::CommandQueue command_queue);
  ~OpenCLRuntime();
  OpenCLRuntime(const OpenCLRuntime&) = delete;
  OpenCLRuntime &operator=(const OpenCLRuntime&) = delete;

  bool BuildProgram(const std::string &kernel_name,
                    const std::string &build_options,
                    cl::Program *program);

 private:
  cl::Context context_;
  cl::Device device_;
  cl::CommandQueue command_queue_;
  cl::Program program_;
  std::once_flag build_flag_;
  std::string kernel_path_;
  static const std::unordered_map<std::string,
               std::string> kernel_program_map_;
  mutable std::unordered_map<std::string,
          cl::Program> built_program_map_;
};

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
