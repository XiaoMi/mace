//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
#define MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_

#include <map>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <set>
#include <string>

#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_wrapper.h"
#include "mace/utils/timer.h"

namespace mace {

class OpenCLProfilingTimer : public Timer {
 public:
  explicit OpenCLProfilingTimer(const cl::Event *event)
      : event_(event), accumulated_micros_(0) {}
  void StartTiming() override;
  void StopTiming() override;
  void AccumulateTiming() override;
  void ClearTiming() override;
  double ElapsedMicros() override;
  double AccumulatedMicros() override;

 private:
  const cl::Event *event_;
  double start_nanos_;
  double stop_nanos_;
  double accumulated_micros_;
};

class OpenCLRuntime {
 public:
  static OpenCLRuntime *Global();
  static OpenCLRuntime *CreateGlobal(GPUType, GPUPerfHint, GPUPriorityHint);

  cl::Context &context();
  cl::Device &device();
  cl::CommandQueue &command_queue();

  void GetCallStats(const cl::Event &event, CallStats *stats);
  uint32_t GetDeviceMaxWorkGroupSize();
  uint32_t GetKernelMaxWorkGroupSize(const cl::Kernel &kernel);
  cl::Kernel BuildKernel(const std::string &program_name,
                         const std::string &kernel_name,
                         const std::set<std::string> &build_options);

 private:
  OpenCLRuntime(GPUType, GPUPerfHint, GPUPriorityHint);
  ~OpenCLRuntime();
  OpenCLRuntime(const OpenCLRuntime &) = delete;
  OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

  void BuildProgram(const std::string &program_file_name,
                    const std::string &binary_file_name,
                    const std::string &build_options,
                    cl::Program *program);
  std::string GenerateCLBinaryFilenamePrefix(const std::string &filename_msg);

 private:
  // All OpenCL object must be a pointer and manually deleted before unloading
  // OpenCL library.
  cl::Context *context_;
  cl::Device *device_;
  cl::CommandQueue *command_queue_;
  std::map<std::string, cl::Program> built_program_map_;
  std::mutex program_build_mutex_;
  std::string kernel_path_;
};

static OpenCLRuntime *opencl_runtime_instance = nullptr;
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
