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

#ifndef MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
#define MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_

#include <map>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <set>
#include <string>
#include <vector>

#include "mace/core/kv_storage.h"
#include "mace/core/future.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/scratch_image.h"
#include "mace/proto/mace.pb.h"
#include "mace/utils/string_util.h"
#include "mace/utils/timer.h"
#include "mace/utils/tuner.h"

namespace mace {

enum GPUType {
  QUALCOMM_ADRENO,
  MALI,
  PowerVR,
  UNKNOWN,
};

enum OpenCLVersion {
  CL_VER_1_0,
  CL_VER_1_1,
  CL_VER_1_2,
  CL_VER_2_0,
  CL_VER_UNKNOWN,
};


const std::string OpenCLErrorToString(cl_int error);

#define MACE_CL_RET_ERROR(error)                            \
  if (error != CL_SUCCESS) {                                \
    LOG(ERROR) << "error: " << OpenCLErrorToString(error);  \
    return error;                                           \
  }

#define MACE_CL_RET_STATUS(error)                           \
  if (error != CL_SUCCESS) {                                \
    LOG(ERROR) << "error: " << OpenCLErrorToString(error);  \
    return MaceStatus::MACE_OUT_OF_RESOURCES;               \
  }

class OpenCLRuntime {
 public:
  OpenCLRuntime(
      std::shared_ptr<KVStorage> cache_storage = nullptr,
      const GPUPriorityHint priority_hint = GPUPriorityHint::PRIORITY_NORMAL,
      const GPUPerfHint perf_hint = GPUPerfHint::PERF_NORMAL,
      std::shared_ptr<KVStorage> precompiled_binary_storage = nullptr,
      std::shared_ptr<Tuner<uint32_t>> tuner = nullptr);
  ~OpenCLRuntime();
  OpenCLRuntime(const OpenCLRuntime &) = delete;
  OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

  cl::Context &context();
  cl::Device &device();
  cl::CommandQueue &command_queue();
  GPUType gpu_type() const;
  const std::string platform_info() const;
  uint64_t device_global_mem_cache_size() const;
  uint32_t device_compute_units() const;
  Tuner<uint32_t> *tuner();
  bool is_opencl_avaliable();

  void GetCallStats(const cl::Event &event, CallStats *stats);
  uint64_t GetDeviceMaxWorkGroupSize();
  uint64_t GetDeviceMaxMemAllocSize();
  bool IsImageSupport();
  std::vector<uint64_t> GetMaxImage2DSize();
  uint64_t GetKernelMaxWorkGroupSize(const cl::Kernel &kernel);
  uint64_t GetKernelWaveSize(const cl::Kernel &kernel);
  bool IsNonUniformWorkgroupsSupported() const;
  bool IsOutOfRangeCheckEnabled() const;
  bool is_profiling_enabled() const;

  MaceStatus BuildKernel(const std::string &program_name,
                         const std::string &kernel_name,
                         const std::set<std::string> &build_options,
                         cl::Kernel *kernel);

  void SaveBuiltCLProgram();

 private:
  bool BuildProgram(const std::string &program_file_name,
                    const std::string &binary_file_name,
                    const std::string &build_options,
                    cl::Program *program);
  bool BuildProgramFromCache(
      const std::string &built_program_key,
      const std::string &build_options_str,
      cl::Program *program);
  bool BuildProgramFromPrecompiledBinary(
      const std::string &built_program_key,
      const std::string &build_options_str,
      cl::Program *program);
  bool BuildProgramFromSource(
      const std::string &program_name,
      const std::string &built_program_key,
      const std::string &build_options_str,
      cl::Program *program);
  OpenCLVersion ParseDeviceVersion(const std::string &device_version);

 private:
  std::shared_ptr<KVStorage> cache_storage_;
  std::shared_ptr<KVStorage> precompiled_binary_storage_;
  std::shared_ptr<Tuner<uint32_t>> tuner_;
  bool is_opencl_avaliable_;
  bool is_profiling_enabled_;
  OpenCLVersion opencl_version_;
  GPUType gpu_type_;
  // All OpenCL object must be a pointer and manually deleted before unloading
  // OpenCL library.
  std::shared_ptr<cl::Context> context_;
  std::shared_ptr<cl::Device> device_;
  std::shared_ptr<cl::CommandQueue> command_queue_;
  std::map<std::string, cl::Program> built_program_map_;
  std::mutex program_build_mutex_;
  std::string platform_info_;
  std::string precompiled_binary_platform_info_;
  bool out_of_range_check_;
  uint64_t device_global_mem_cache_size_;
  uint32_t device_compute_units_;
};

class OpenCLProfilingTimer : public Timer {
 public:
  OpenCLProfilingTimer(OpenCLRuntime *runtime, const cl::Event *event)
      : runtime_(runtime), event_(event), accumulated_micros_(0) {}
  void StartTiming() override;
  void StopTiming() override;
  void AccumulateTiming() override;
  void ClearTiming() override;
  double ElapsedMicros() override;
  double AccumulatedMicros() override;

 private:
  OpenCLRuntime *runtime_;
  const cl::Event *event_;
  double start_nanos_;
  double stop_nanos_;
  double accumulated_micros_;
};
}  // namespace mace

#endif  // MACE_CORE_RUNTIME_OPENCL_OPENCL_RUNTIME_H_
