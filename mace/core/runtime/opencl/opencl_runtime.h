//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
#define MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_

#include <map>
#include <memory>
#include <mutex>
#include <set>

#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_wrapper.h"

namespace mace {

class OpenCLRuntime {
 public:
  static OpenCLRuntime *Get();

  static void EnableProfiling();
  cl::Event *GetDefaultEvent();

  cl_ulong GetEventProfilingStartInfo();
  cl_ulong GetEventProfilingEndInfo();


  cl::Context &context();
  cl::Device &device();
  cl::CommandQueue &command_queue();
  cl::Program &program();

  uint32_t GetDeviceMaxWorkGroupSize();
  uint32_t GetKernelMaxWorkGroupSize(const cl::Kernel& kernel);
  cl::Kernel BuildKernel(const std::string &program_name,
                         const std::string &kernel_name,
                         const std::set<std::string> &build_options);
 private:
  OpenCLRuntime(cl::Context context,
                cl::Device device,
                cl::CommandQueue command_queue);
  ~OpenCLRuntime();
  OpenCLRuntime(const OpenCLRuntime&) = delete;
  OpenCLRuntime &operator=(const OpenCLRuntime&) = delete;

  void BuildProgram(const std::string &program_file_name,
                    const std::string &binary_file_name,
                    const std::string &build_options,
                    cl::Program *program);
  std::string GenerateCLBinaryFilenamePrefix(const std::string &filename_msg);

 private:
  static bool enable_profiling_;
  static std::unique_ptr<cl::Event> profiling_ev_;

  cl::Context context_;
  cl::Device device_;
  cl::CommandQueue command_queue_;
  cl::Program program_;
  std::mutex program_build_mutex_;
  std::string kernel_path_;
  static const std::map<std::string,
               std::string> program_map_;
  mutable std::map<std::string,
          cl::Program> built_program_map_;
};

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
