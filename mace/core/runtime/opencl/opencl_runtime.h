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
#include "mace/public/mace_runtime.h"
#include "mace/utils/string_util.h"
#include "mace/utils/timer.h"

namespace mace {

enum GPUType {
  QUALCOMM_ADRENO,
  MALI,
  PowerVR,
  UNKNOWN,
};


const std::string OpenCLErrorToString(cl_int error);

#define MACE_CHECK_CL_SUCCESS(error) \
  MACE_CHECK(error == CL_SUCCESS) << "error: " << OpenCLErrorToString(error)

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
  static void Configure(GPUPerfHint, GPUPriorityHint);

  cl::Context &context();
  cl::Device &device();
  cl::CommandQueue &command_queue();

  void GetCallStats(const cl::Event &event, CallStats *stats);
  uint64_t GetDeviceMaxWorkGroupSize();
  uint64_t GetKernelMaxWorkGroupSize(const cl::Kernel &kernel);
  uint64_t GetKernelWaveSize(const cl::Kernel &kernel);
  const bool IsNonUniformWorkgroupsSupported();
  const GPUType ParseGPUTypeFromDeviceName(const std::string &device_name);
  const GPUType gpu_type() const;
  cl::Kernel BuildKernel(const std::string &program_name,
                         const std::string &kernel_name,
                         const std::set<std::string> &build_options);
  const bool IsOutOfRangeCheckEnabled() const;

 private:
  OpenCLRuntime(GPUPerfHint, GPUPriorityHint);
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
  std::shared_ptr<cl::Context> context_;
  std::shared_ptr<cl::Device> device_;
  std::shared_ptr<cl::CommandQueue> command_queue_;
  std::map<std::string, cl::Program> built_program_map_;
  std::mutex program_build_mutex_;
  std::string kernel_path_;
  GPUType gpu_type_;
  std::string opencl_version_;
  bool out_of_range_check_;

  static GPUPerfHint gpu_perf_hint_;
  static GPUPriorityHint gpu_priority_hint_;
};

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
