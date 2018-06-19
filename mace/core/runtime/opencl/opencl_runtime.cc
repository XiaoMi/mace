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

#include "mace/core/runtime/opencl/opencl_runtime.h"

#include <sys/stat.h>

#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <vector>
#include <utility>

#include "mace/public/mace_runtime.h"
#include "mace/core/macros.h"
#include "mace/core/file_storage.h"
#include "mace/core/runtime/opencl/opencl_extension.h"
#include "mace/public/mace.h"
#include "mace/utils/tuner.h"

namespace mace {

extern const std::map<std::string, std::vector<unsigned char>>
    kEncryptedProgramMap;

void SetGPUHints(GPUPerfHint gpu_perf_hint, GPUPriorityHint gpu_priority_hint) {
  VLOG(1) << "Set GPU configurations, gpu_perf_hint: " << gpu_perf_hint
          << ", gpu_priority_hint: " << gpu_priority_hint;
  OpenCLRuntime::Configure(gpu_perf_hint, gpu_priority_hint);
}

// Set OpenCL Compiled Binary paths, just call once. (Not thread-safe)
void SetOpenCLBinaryPaths(const std::vector<std::string> &paths) {
  OpenCLRuntime::ConfigureOpenCLBinaryPath(paths);
}


const std::string OpenCLErrorToString(cl_int error) {
  switch (error) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:
      return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
      return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case CL_COMPILE_PROGRAM_FAILURE:
      return "CL_COMPILE_PROGRAM_FAILURE";
    case CL_LINKER_NOT_AVAILABLE:
      return "CL_LINKER_NOT_AVAILABLE";
    case CL_LINK_PROGRAM_FAILURE:
      return "CL_LINK_PROGRAM_FAILURE";
    case CL_DEVICE_PARTITION_FAILED:
      return "CL_DEVICE_PARTITION_FAILED";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
      return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_INVALID_PROPERTY:
      return "CL_INVALID_PROPERTY";
    case CL_INVALID_IMAGE_DESCRIPTOR:
      return "CL_INVALID_IMAGE_DESCRIPTOR";
    case CL_INVALID_COMPILER_OPTIONS:
      return "CL_INVALID_COMPILER_OPTIONS";
    case CL_INVALID_LINKER_OPTIONS:
      return "CL_INVALID_LINKER_OPTIONS";
    case CL_INVALID_DEVICE_PARTITION_COUNT:
      return "CL_INVALID_DEVICE_PARTITION_COUNT";
    case CL_INVALID_PIPE_SIZE:
      return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE:
      return "CL_INVALID_DEVICE_QUEUE";
    default:
      return MakeString("UNKNOWN: ", error);
  }
}

namespace {
void OpenCLPrintfCallback(const char *buffer,
                          size_t length,
                          size_t final,
                          void *user_data) {
  MACE_UNUSED(final);
  MACE_UNUSED(user_data);
  fwrite(buffer, 1, length, stdout);
}

void GetAdrenoContextProperties(std::vector<cl_context_properties> *properties,
                                GPUPerfHint gpu_perf_hint,
                                GPUPriorityHint gpu_priority_hint) {
  MACE_CHECK_NOTNULL(properties);
  switch (gpu_perf_hint) {
    case GPUPerfHint::PERF_LOW:
      properties->push_back(CL_CONTEXT_PERF_HINT_QCOM);
      properties->push_back(CL_PERF_HINT_LOW_QCOM);
      break;
    case GPUPerfHint::PERF_NORMAL:
      properties->push_back(CL_CONTEXT_PERF_HINT_QCOM);
      properties->push_back(CL_PERF_HINT_NORMAL_QCOM);
      break;
    case GPUPerfHint::PERF_HIGH:
      properties->push_back(CL_CONTEXT_PERF_HINT_QCOM);
      properties->push_back(CL_PERF_HINT_HIGH_QCOM);
      break;
    default:
      break;
  }
  switch (gpu_priority_hint) {
    case GPUPriorityHint::PRIORITY_LOW:
      properties->push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
      properties->push_back(CL_PRIORITY_HINT_LOW_QCOM);
      break;
    case GPUPriorityHint::PRIORITY_NORMAL:
      properties->push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
      properties->push_back(CL_PRIORITY_HINT_NORMAL_QCOM);
      break;
    case GPUPriorityHint::PRIORITY_HIGH:
      properties->push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
      properties->push_back(CL_PRIORITY_HINT_HIGH_QCOM);
      break;
    default:
      break;
  }
  // The properties list should be terminated with 0
  properties->push_back(0);
}

GPUType ParseGPUType(const std::string &device_name) {
  constexpr const char *kQualcommAdrenoGPUStr = "QUALCOMM Adreno(TM)";
  constexpr const char *kMaliGPUStr = "Mali";
  constexpr const char *kPowerVRGPUStr = "PowerVR";

  if (device_name == kQualcommAdrenoGPUStr) {
    return GPUType::QUALCOMM_ADRENO;
  } else if (device_name.find(kMaliGPUStr) != std::string::npos) {
    return GPUType::MALI;
  } else if (device_name.find(kPowerVRGPUStr) != std::string::npos) {
    return GPUType::PowerVR;
  } else {
    return GPUType::UNKNOWN;
  }
}

std::string FindFirstExistPath(const std::vector<std::string> &paths) {
  std::string result;
  struct stat st;
  for (auto path : paths) {
    if (stat(path.c_str(), &st) == 0) {
      if (S_ISREG(st.st_mode)) {
        result = path;
        break;
      }
    }
  }
  return result;
}

const char *kOpenCLPlatformInfoKey =
    "mace_opencl_precompiled_platform_info_key";
const char *kPrecompiledProgramFileName =
    "mace_cl_compiled_program.bin";
}  // namespace

void OpenCLProfilingTimer::StartTiming() {}

void OpenCLProfilingTimer::StopTiming() {
  OpenCLRuntime::Global()->command_queue().finish();
  start_nanos_ = event_->getProfilingInfo<CL_PROFILING_COMMAND_START>();
  stop_nanos_ = event_->getProfilingInfo<CL_PROFILING_COMMAND_END>();
}

double OpenCLProfilingTimer::ElapsedMicros() {
  return (stop_nanos_ - start_nanos_) / 1000.0;
}

double OpenCLProfilingTimer::AccumulatedMicros() { return accumulated_micros_; }

void OpenCLProfilingTimer::AccumulateTiming() {
  StopTiming();
  accumulated_micros_ += (stop_nanos_ - start_nanos_) / 1000.0;
}

void OpenCLProfilingTimer::ClearTiming() {
  start_nanos_ = 0;
  stop_nanos_ = 0;
  accumulated_micros_ = 0;
}

GPUPerfHint OpenCLRuntime::kGPUPerfHint = GPUPerfHint::PERF_NORMAL;
GPUPriorityHint OpenCLRuntime::kGPUPriorityHint =
    GPUPriorityHint::PRIORITY_DEFAULT;
std::string
    OpenCLRuntime::kPrecompiledBinaryPath = "";  // NOLINT(runtime/string)

OpenCLRuntime *OpenCLRuntime::Global() {
  static OpenCLRuntime runtime;
  return &runtime;
}

void OpenCLRuntime::Configure(GPUPerfHint gpu_perf_hint,
                              GPUPriorityHint gpu_priority_hint) {
  OpenCLRuntime::kGPUPerfHint = gpu_perf_hint;
  OpenCLRuntime::kGPUPriorityHint = gpu_priority_hint;
}

void OpenCLRuntime::ConfigureOpenCLBinaryPath(
    const std::vector<std::string> &paths) {
  OpenCLRuntime::kPrecompiledBinaryPath = FindFirstExistPath(paths);
  if (OpenCLRuntime::kPrecompiledBinaryPath.empty()) {
    LOG(WARNING) << "There is no precompiled OpenCL binary file in "
                 << MakeString(paths);
  }
}

OpenCLRuntime::OpenCLRuntime():
    precompiled_binary_storage_(nullptr),
    cache_storage_(nullptr),
    is_profiling_enabled_(false) {
  LoadOpenCLLibrary();

  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
    LOG(FATAL) << "No OpenCL platforms found";
  }
  cl::Platform default_platform = all_platforms[0];
  std::stringstream ss;
  ss << default_platform.getInfo<CL_PLATFORM_NAME>()
     << ", " << default_platform.getInfo<CL_PLATFORM_PROFILE>() << ", "
     << default_platform.getInfo<CL_PLATFORM_VERSION>();
  platform_info_ = ss.str();
  VLOG(1) << "Using platform: " << platform_info_;

  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if (all_devices.size() == 0) {
    LOG(FATAL) << "No OpenCL devices found";
  }

  bool gpu_detected = false;
  device_ = std::make_shared<cl::Device>();
  for (auto device : all_devices) {
    if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
      *device_ = device;
      gpu_detected = true;

      const std::string device_name = device.getInfo<CL_DEVICE_NAME>();
      gpu_type_ = ParseGPUType(device_name);

      const std::string device_version = device.getInfo<CL_DEVICE_VERSION>();
      opencl_version_ = ParseDeviceVersion(device_version);

      VLOG(1) << "Using device: " << device_name;
      break;
    }
  }
  if (!gpu_detected) {
    LOG(FATAL) << "No GPU device found";
  }

  cl_command_queue_properties properties = 0;

  const char *profiling = getenv("MACE_OPENCL_PROFILING");
  if (Tuner<uint32_t>::Get()->IsTuning() ||
      (profiling != nullptr && strlen(profiling) == 1 && profiling[0] == '1')) {
    properties |= CL_QUEUE_PROFILING_ENABLE;
    is_profiling_enabled_ = true;
  }

  cl_int err;
  if (gpu_type_ == GPUType::QUALCOMM_ADRENO
          && opencl_version_ == OpenCLVersion::CL_VER_2_0) {
    std::vector<cl_context_properties> context_properties;
    context_properties.reserve(5);
    GetAdrenoContextProperties(&context_properties,
                               OpenCLRuntime::kGPUPerfHint,
                               OpenCLRuntime::kGPUPriorityHint);
    context_ = std::shared_ptr<cl::Context>(
        new cl::Context({*device_}, context_properties.data(),
                        nullptr, nullptr, &err));
  } else {
    if (is_profiling_enabled_ && gpu_type_ == GPUType::MALI) {
      std::vector<cl_context_properties> context_properties = {
          CL_CONTEXT_PLATFORM, (cl_context_properties)default_platform(),
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)OpenCLPrintfCallback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000, 0
      };
      context_ = std::shared_ptr<cl::Context>(
          new cl::Context({*device_}, context_properties.data(),
                          nullptr, nullptr, &err));
    } else {
      context_ = std::shared_ptr<cl::Context>(
          new cl::Context({*device_}, nullptr, nullptr, nullptr, &err));
    }
  }
  MACE_CHECK_CL_SUCCESS(err);

  command_queue_ = std::make_shared<cl::CommandQueue>(*context_,
                                                      *device_,
                                                      properties,
                                                      &err);
  MACE_CHECK_CL_SUCCESS(err);

  extern std::shared_ptr<KVStorageFactory> kStorageFactory;
  if (kStorageFactory != nullptr) {
    cache_storage_ =
        kStorageFactory->CreateStorage(kPrecompiledProgramFileName);

    if (cache_storage_->Load() != 0) {
      LOG(WARNING) << "Load OpenCL cached compiled kernel file failed. "
                   << "Please make sure the storage directory exist "
                   << "and you have Write&Read permission";
    }
    auto platform_info_array =
        this->cache_storage_->Find(kOpenCLPlatformInfoKey);
    if (platform_info_array != nullptr) {
      cached_binary_platform_info_ =
          std::string(platform_info_array->begin(),
                      platform_info_array->end());
    }
  }

  if (cached_binary_platform_info_ != platform_info_) {
    if (OpenCLRuntime::kPrecompiledBinaryPath.empty()) {
      LOG(WARNING) << "There is no precompiled OpenCL binary in"
          " all OpenCL binary paths";
    } else {
      precompiled_binary_storage_.reset(
          new FileStorage(OpenCLRuntime::kPrecompiledBinaryPath));
      if (precompiled_binary_storage_->Load() != 0) {
        LOG(WARNING) << "Load OpenCL precompiled kernel file failed. "
                     << "Please make sure the storage directory exist "
                     << "and you have Write&Read permission";
      }

      auto platform_info_array =
          this->precompiled_binary_storage_->Find(kOpenCLPlatformInfoKey);
      if (platform_info_array != nullptr) {
        precompiled_binary_platform_info_ =
            std::string(platform_info_array->begin(),
                        platform_info_array->end());
      }
    }
  }

  device_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                   &device_gloabl_mem_cache_size_);

  device_->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,
                   &device_compute_units_);
  const char *out_of_range_check = getenv("MACE_OUT_OF_RANGE_CHECK");
  if (out_of_range_check != nullptr && strlen(out_of_range_check) == 1
      && out_of_range_check[0] == '1') {
    this->out_of_range_check_ = true;
  } else {
    this->out_of_range_check_ = false;
  }
}

OpenCLRuntime::~OpenCLRuntime() {
  built_program_map_.clear();
  // We need to control the destruction order, which has dependencies
  command_queue_.reset();
  context_.reset();
  device_.reset();
  UnloadOpenCLLibrary();
}

cl::Context &OpenCLRuntime::context() { return *context_; }

cl::Device &OpenCLRuntime::device() { return *device_; }

cl::CommandQueue &OpenCLRuntime::command_queue() { return *command_queue_; }

uint64_t OpenCLRuntime::device_global_mem_cache_size() const {
  return device_gloabl_mem_cache_size_;
}

uint32_t OpenCLRuntime::device_compute_units() const {
  return device_compute_units_;
}

bool OpenCLRuntime::BuildProgramFromCache(
    const std::string &built_program_key,
    const std::string &build_options_str,
    cl::Program *program) {
  // Find from binary
  if (this->cache_storage_ == nullptr) return false;
  if (cached_binary_platform_info_ != platform_info_) {
    VLOG(3) << "cached OpenCL binary version is not same"
        " with current version";
    return false;
  }
  auto content = this->cache_storage_->Find(built_program_key);
  if (content == nullptr) {
    return false;
  }

  *program = cl::Program(context(), {device()}, {*content});
  cl_int ret = program->build({device()}, build_options_str.c_str());
  if (ret != CL_SUCCESS) {
    if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
        CL_BUILD_ERROR) {
      std::string build_log =
          program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
      LOG(INFO) << "Program build log: " << build_log;
    }
    LOG(WARNING) << "Build program "
                 << built_program_key << " from Cache failed:"
                 << MakeString(ret);
    return false;
  }
  VLOG(3) << "Program from Cache: " << built_program_key;
  return true;
}

bool OpenCLRuntime::BuildProgramFromPrecompiledBinary(
    const std::string &built_program_key,
    const std::string &build_options_str,
    cl::Program *program) {
  // Find from binary
  if (this->precompiled_binary_storage_ == nullptr) return false;
  if (precompiled_binary_platform_info_ != platform_info_) {
    VLOG(3) << "precompiled OpenCL binary version "
            << precompiled_binary_platform_info_
            << " is not same with current version";
    return false;
  }
  auto content = this->precompiled_binary_storage_->Find(built_program_key);
  if (content == nullptr) {
    return false;
  }

  *program = cl::Program(context(), {device()}, {*content});
  cl_int ret = program->build({device()}, build_options_str.c_str());
  if (ret != CL_SUCCESS) {
    if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
        CL_BUILD_ERROR) {
      std::string build_log =
          program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
      LOG(INFO) << "Program build log: " << build_log;
    }
    LOG(WARNING) << "Build program "
                 << built_program_key << " from precompiled binary failed:"
                 << MakeString(ret);
    return false;
  }
  VLOG(3) << "Program from precompiled binary: " << built_program_key;
  return true;
}

void OpenCLRuntime::BuildProgramFromSource(
    const std::string &program_name,
    const std::string &built_program_key,
    const std::string &build_options_str,
    cl::Program *program) {
  // Find from source
  auto it_source = kEncryptedProgramMap.find(program_name);
  if (it_source != kEncryptedProgramMap.end()) {
    cl::Program::Sources sources;
    std::string source(it_source->second.begin(), it_source->second.end());
    std::string kernel_source = ObfuscateString(source);
    sources.push_back(kernel_source);
    *program = cl::Program(context(), sources);
    cl_int ret = program->build({device()}, build_options_str.c_str());
    if (ret != CL_SUCCESS) {
      if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
          CL_BUILD_ERROR) {
        std::string build_log =
            program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
        LOG(INFO) << "Program build log: " << build_log;
      }
      LOG(WARNING) << "Build program "
                   << program_name << " from source failed: "
                   << MakeString(ret);
      return;
    }

    // Keep built program binary
    size_t device_list_size = 1;
    std::unique_ptr<size_t[]> program_binary_sizes(
        new size_t[device_list_size]);
    cl_int err = clGetProgramInfo((*program)(), CL_PROGRAM_BINARY_SIZES,
                                  sizeof(size_t) * device_list_size,
                                  program_binary_sizes.get(), nullptr);
    MACE_CHECK_CL_SUCCESS(err);
    std::unique_ptr<std::unique_ptr<unsigned char[]>[]> program_binaries(
        new std::unique_ptr<unsigned char[]>[device_list_size]);
    for (cl_uint i = 0; i < device_list_size; ++i) {
      program_binaries[i] = std::unique_ptr<unsigned char[]>(
          new unsigned char[program_binary_sizes[i]]);
    }

    err = clGetProgramInfo((*program)(), CL_PROGRAM_BINARIES,
                           sizeof(unsigned char *) * device_list_size,
                           program_binaries.get(), nullptr);
    MACE_CHECK_CL_SUCCESS(err);
    std::vector<unsigned char> content(
        reinterpret_cast<unsigned char const *>(program_binaries[0].get()),
        reinterpret_cast<unsigned char const *>(program_binaries[0].get()) +
            program_binary_sizes[0]);

    if (this->cache_storage_ != nullptr) {
      this->cache_storage_->Insert(built_program_key, content);
      // update platform info
      this->cache_storage_->Insert(
          kOpenCLPlatformInfoKey,
          std::vector<unsigned char>(platform_info_.begin(),
                                     platform_info_.end()));
    }

    VLOG(3) << "Program from source: " << built_program_key;
  }
}

void OpenCLRuntime::BuildProgram(const std::string &program_name,
                                 const std::string &built_program_key,
                                 const std::string &build_options,
                                 cl::Program *program) {
  MACE_CHECK_NOTNULL(program);

  std::string build_options_str =
      build_options + " -Werror -cl-mad-enable -cl-fast-relaxed-math";
  // Build flow: cache -> precompiled binary -> source
  bool ret = BuildProgramFromCache(built_program_key,
                                   build_options_str, program);
  if (!ret) {
    ret = BuildProgramFromPrecompiledBinary(built_program_key,
                                            build_options_str, program);
    if (!ret) {
      BuildProgramFromSource(program_name, built_program_key,
                             build_options_str, program);
    }
  }
}

cl::Kernel OpenCLRuntime::BuildKernel(
    const std::string &program_name,
    const std::string &kernel_name,
    const std::set<std::string> &build_options) {
  std::string build_options_str;
  for (auto &option : build_options) {
    build_options_str += " " + option;
  }
  std::string built_program_key = program_name + build_options_str;

  std::lock_guard<std::mutex> lock(program_build_mutex_);
  auto built_program_it = built_program_map_.find(built_program_key);
  cl::Program program;
  if (built_program_it != built_program_map_.end()) {
    program = built_program_it->second;
  } else {
    this->BuildProgram(program_name, built_program_key, build_options_str,
                       &program);
    built_program_map_.emplace(built_program_key, program);
  }
  return cl::Kernel(program, kernel_name.c_str());
}

void OpenCLRuntime::SaveBuiltCLProgram() {
  if (cache_storage_ != nullptr) {
    if (cache_storage_->Flush() != 0) {
      LOG(FATAL) << "Store OPENCL compiled kernel to file failed. "
                 << "Please make sure the storage directory exist "
                 << "and you have Write&Read permission";
    }
  }
}

void OpenCLRuntime::GetCallStats(const cl::Event &event, CallStats *stats) {
  if (stats != nullptr) {
    stats->start_micros =
        event.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
    stats->end_micros =
        event.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;
  }
}

uint64_t OpenCLRuntime::GetDeviceMaxWorkGroupSize() {
  uint64_t size = 0;
  device_->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
  return size;
}

uint64_t OpenCLRuntime::GetKernelMaxWorkGroupSize(const cl::Kernel &kernel) {
  uint64_t size = 0;
  kernel.getWorkGroupInfo(*device_, CL_KERNEL_WORK_GROUP_SIZE, &size);
  return size;
}

uint64_t OpenCLRuntime::GetKernelWaveSize(const cl::Kernel &kernel) {
  uint64_t size = 0;
  kernel.getWorkGroupInfo(*device_, CL_KERNEL_WAVE_SIZE_QCOM, &size);
  return size;
}

bool OpenCLRuntime::IsNonUniformWorkgroupsSupported() const {
  return (gpu_type_ == GPUType::QUALCOMM_ADRENO &&
      opencl_version_ == OpenCLVersion::CL_VER_2_0);
}

GPUType OpenCLRuntime::gpu_type() const {
  return gpu_type_;
}

const std::string OpenCLRuntime::platform_info() const {
  return platform_info_;
}

OpenCLVersion OpenCLRuntime::ParseDeviceVersion(
    const std::string &device_version) {
  // OpenCL Device version string format:
  // OpenCL<space><major_version.minor_version><space>
  // <vendor-specific information>
  auto words = Split(device_version, ' ');
  if (words[1] == "2.0") {
    return OpenCLVersion::CL_VER_2_0;
  } else if (words[1] == "1.2") {
    return OpenCLVersion::CL_VER_1_2;
  } else if (words[1] == "1.1") {
    return OpenCLVersion::CL_VER_1_1;
  } else if (words[1] == "1.0") {
    return OpenCLVersion::CL_VER_1_0;
  } else {
    LOG(FATAL) << "Do not support OpenCL version: " << words[1];
    return OpenCLVersion::CL_VER_1_0;
  }
}

bool OpenCLRuntime::IsOutOfRangeCheckEnabled() const {
  return out_of_range_check_;
}

bool OpenCLRuntime::is_profiling_enabled() const {
  return is_profiling_enabled_;
}

}  // namespace mace
