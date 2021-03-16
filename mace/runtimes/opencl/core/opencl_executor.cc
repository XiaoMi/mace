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

#include "mace/runtimes/opencl/core/opencl_executor.h"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include "mace/codegen/opencl/encrypt_opencl_kernel.h"
#include "mace/core/kv_storage.h"
#include "mace/runtimes/opencl/core/opencl_extension.h"
#include "mace/utils/macros.h"
#include "mace/utils/tuner.h"

namespace mace {

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
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    case CL_INVALID_PIPE_SIZE:
      return "CL_INVALID_PIPE_SIZE";
    case CL_INVALID_DEVICE_QUEUE:
      return "CL_INVALID_DEVICE_QUEUE";
#endif
    default:
      return MakeString("UNKNOWN: ", error);
  }
}

namespace {
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
void OpenCLPrintfCallback(const char *buffer,
                          size_t length,
                          size_t final,
                          void *user_data) {
  MACE_UNUSED(final);
  MACE_UNUSED(user_data);
  fwrite(buffer, 1, length, stdout);
}
#endif

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

cl::Platform FindGpuPlatform() {
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  MACE_CHECK(!all_platforms.empty(), "No OpenCL platforms found");
  return all_platforms[0];
}

std::vector<unsigned char> GetBinaryFromProgram(const cl::Program &program) {
  // Keep built program binary
  size_t device_list_size = 1;
  std::unique_ptr<size_t[]> program_binary_sizes(
      new size_t[device_list_size]);
  cl_int err = clGetProgramInfo(program(), CL_PROGRAM_BINARY_SIZES,
                                sizeof(size_t) * device_list_size,
                                program_binary_sizes.get(), nullptr);
  MACE_CHECK(err == CL_SUCCESS, "error: ", OpenCLErrorToString(err));

  std::unique_ptr<std::unique_ptr<unsigned char[]>[]> program_binaries(
      new std::unique_ptr<unsigned char[]>[device_list_size]);
  for (cl_uint i = 0; i < device_list_size; ++i) {
    program_binaries[i] = std::unique_ptr<unsigned char[]>(
        new unsigned char[program_binary_sizes[i]]);
  }

  err = clGetProgramInfo(program(), CL_PROGRAM_BINARIES,
                         sizeof(unsigned char *) * device_list_size,
                         program_binaries.get(), nullptr);
  MACE_CHECK(err == CL_SUCCESS, "error: ", OpenCLErrorToString(err));

  std::vector<unsigned char> content(
      reinterpret_cast<unsigned char const *>(program_binaries[0].get()),
      reinterpret_cast<unsigned char const *>(program_binaries[0].get()) +
          program_binary_sizes[0]);

  return content;
}

std::shared_ptr<cl::Device> FindGpuDevice(
    const cl::Platform &default_platform) {
  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  MACE_CHECK(!all_devices.empty(), "No OpenCL devices found");

  for (auto device : all_devices) {
    if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
      return std::make_shared<cl::Device>(device);
    }
  }

  LOG(ERROR) << "No GPU device found";
  return nullptr;
}

const char *kOpenCLPlatformInfoKey =
    "mace_opencl_precompiled_platform_info_key";

const char *kOpenCLDeviceNameKey =
    "mace_opencl_precompiled_device_name_key";
}  // namespace

void OpenCLProfilingTimer::StartTiming() {}

void OpenCLProfilingTimer::StopTiming() {
  runtime_->command_queue().finish();
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

std::string ParseQcomRemoteBranchFromVersion(const std::string &version) {
  std::string pattern("Remote Branch:");
  return GetStrAfterPattern(version, pattern);
}

/// Parse build date from OpenCL version string generated by Qualcomm platform.
///
/// \param version Version string returned by CL_PLATFORM_VERSION.
///        See the following note for detailed format of version string.
/// \return For legal input: 3-vector which contains year, date and moth.
///         For illegal input: empty vector.
///
/// Note:
/// 1. Format of version string(for Qualcomm platform only):
///    OpenCL<space><major_version.minor_version><space>vendor_name<space>
///    build:<space>build_info<space>
///    Date:<space><month>/<day_of_month>/<year><space><day_of_week><space>
///    Local Branch:<space><local_branch_name><space>
///    Remote Branch:<space><remote_branch_name>
///
/// 2. One or two of local_branch_name and remote_branch_name may be empty
///    in some cases.
///
/// 3. <month>, <day_of_month> and <year> contain 2 characters currently,
///    but <year> may contain 4 characters in the future.
std::vector<int> ParseQcomDateFromVersion(const std::string &version) {
  std::string date_prefix("Date:");
  std::string date_str = GetStrAfterPattern(version, date_prefix);
  if (date_str.empty()) {
    return std::vector<int>();
  }
  std::vector<std::string> month_date_year = Split(date_str, '/');
  if (month_date_year.size() != 3) {
    return std::vector<int>();
  }
  if (month_date_year[2].size() == 2) {
    month_date_year[2] = "20" + month_date_year[2];
  }
  std::vector<int> date_vec(3);
  date_vec[0] = stoi(month_date_year[2]);  // year
  date_vec[1] = stoi(month_date_year[0]);  // month
  date_vec[2] = stoi(month_date_year[1]);  // date
  return date_vec;
}

bool DateNewerOrEqual(const std::vector<int> &date,
                      const std::vector<int> &cached_date) {
  // date and cached_date should be 3-vectors
  if (date.size() != 3 || cached_date.size() != 3) return false;
  for (int i = 0; i < 3; ++i) {
    if (date[i] < cached_date[i]) {
      return false;
    } else if (date[i] > cached_date[i]) {
      return true;
    }
  }
  return true;
}

/// Policy for checking compatibility:
/// 1. For non-Adreno GPU, current OpenCL can use cached OpenCL bin file only
///    when their OpenCL version strings are the same.
/// 2. For Adreno GPU, there are two cases:
/// 2.1 When remote branch name in current OpenCL version string and cached
///     OpenCL version string are the same, current OpenCL can use cached
///     OpenCL bin file no matter their version;
/// 2.2 When remote branch name is not available in current OpenCL version
///     string or cached OpenCL version string, or their remote branch names
///     are not the same, check their build date. Current OpenCL can use
///     cached OpenCL bin file when the build date of current OpenCL is newer
///     than or equal to the build date of cached OpenCL bin file.
bool IsCacheCompatible(const GPUType &gpu_type,
                       const std::string &platform_version,
                       const std::string &cached_binary_platform_info) {
  std::string cached_platform_version = Split(
      cached_binary_platform_info, ',')[2];
  StripString(&cached_platform_version);
  if (gpu_type != QUALCOMM_ADRENO) {
    bool same_version = (platform_version.size() > 0) &&
                        (cached_platform_version.size() > 0) &&
                        (platform_version == cached_platform_version);
    return same_version;
  }
  std::string remote_branch, cached_remote_branch;
  std::vector<int> date, cached_date;
  remote_branch = ParseQcomRemoteBranchFromVersion(platform_version);
  cached_remote_branch = ParseQcomRemoteBranchFromVersion(
      cached_platform_version);
  if ((remote_branch.size() > 0) &&
      (cached_remote_branch.size() > 0) &&
      (remote_branch == cached_remote_branch)) {
    return true;
  } else {
    date = ParseQcomDateFromVersion(platform_version);
    cached_date = ParseQcomDateFromVersion(cached_platform_version);
    if (DateNewerOrEqual(date, cached_date)) {
      return true;
    }
  }
  return false;
}

OpenclExecutor::OpenclExecutor() : is_opencl_avaliable_(false),
                                   is_profiling_enabled_(false),
                                   opencl_version_(CL_VER_UNKNOWN),
                                   gpu_type_(UNKNOWN),
                                   program_key_hash_prefix_("program_hash_ ") {}

MaceStatus OpenclExecutor::Init(std::shared_ptr<OpenclContext> opencl_context,
                                const GPUPriorityHint priority_hint,
                                const GPUPerfHint perf_hint) {
  opencl_context_ = opencl_context;
  auto default_platform = FindGpuPlatform();
  std::stringstream ss;
  std::string platform_version =
      default_platform.getInfo<CL_PLATFORM_VERSION>();
  ss << default_platform.getInfo<CL_PLATFORM_NAME>()
     << ", " << default_platform.getInfo<CL_PLATFORM_PROFILE>() << ", "
     << platform_version << ", "
     << MaceVersion();
  platform_info_ = ss.str();
  VLOG(1) << "Using platform: " << platform_info_;

  device_ = FindGpuDevice(default_platform);
  InitGpuDeviceProperty(*device_);

  cl_command_queue_properties properties = 0;

  const char *profiling = getenv("MACE_OPENCL_PROFILING");
  auto tuner = opencl_context_->opencl_tuner();
  if (tuner->IsTuning() ||
      (profiling != nullptr && strlen(profiling) == 1 && profiling[0] == '1')) {
    properties |= CL_QUEUE_PROFILING_ENABLE;
    is_profiling_enabled_ = true;
  }

  cl_int err;
  if (gpu_type_ == GPUType::QUALCOMM_ADRENO
      && opencl_version_ >= OpenCLVersion::CL_VER_2_0) {
    std::vector<cl_context_properties> context_properties;
    context_properties.reserve(5);
    GetAdrenoContextProperties(&context_properties,
                               perf_hint,
                               priority_hint);
    context_ = std::shared_ptr<cl::Context>(
        new cl::Context({*device_}, context_properties.data(),
                        nullptr, nullptr, &err));
  } else {
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
    if (is_profiling_enabled_ && gpu_type_ == GPUType::MALI) {
      std::vector<cl_context_properties> context_properties = {
          CL_CONTEXT_PLATFORM, (cl_context_properties) default_platform(),
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties) OpenCLPrintfCallback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000, 0
      };
      context_ = std::shared_ptr<cl::Context>(
          new cl::Context({*device_}, context_properties.data(),
                          nullptr, nullptr, &err));
    } else {
      context_ = std::shared_ptr<cl::Context>(
          new cl::Context({*device_}, nullptr, nullptr, nullptr, &err));
    }
#else
    context_ = std::shared_ptr<cl::Context>(
          new cl::Context({*device_}, nullptr, nullptr, nullptr, &err));
#endif
  }
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to create OpenCL Context: "
               << OpenCLErrorToString(err);
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  command_queue_ = std::make_shared<cl::CommandQueue>(*context_,
                                                      *device_,
                                                      properties,
                                                      &err);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to create OpenCL CommandQueue: "
               << OpenCLErrorToString(err);
    return MaceStatus::MACE_OUT_OF_RESOURCES;
  }

  std::string cached_binary_platform_info;
  std::string cached_binary_device_name;
  auto cache_storage = opencl_context_->opencl_cache_storage();
  if (cache_storage != nullptr) {
    if (cache_storage->Load() != 0 && !tuner->IsTuning()) {
      LOG(WARNING) << "Load OpenCL cached compiled kernel file failed. "
                   << "Please make sure the storage directory exist, "
                   << "the file is not modified illegally, "
                   << "and you have Write&Read permission";
    }
    auto platform_info_array = cache_storage->Find(kOpenCLPlatformInfoKey);
    auto *device_name_array = cache_storage->Find(kOpenCLDeviceNameKey);
    if (device_name_array != nullptr) {
      cached_binary_device_name =
          std::string(device_name_array->begin(),
                      device_name_array->end());
    }
    if (platform_info_array != nullptr) {
      cached_binary_platform_info =
          std::string(platform_info_array->begin(),
                      platform_info_array->end());
      bool same_gpu = (device_name_.size() > 0) &&
                            (cached_binary_device_name.size() > 0) &&
                            (device_name_ == cached_binary_device_name);
      bool same_platform_info = (platform_info_.size() > 0) &&
                                (cached_binary_platform_info.size() > 0) &&
                                (platform_info_ == cached_binary_platform_info);
      if (!same_gpu) {
        cache_storage->Clear();
      } else if (!same_platform_info) {
        auto opencl_cache_reuse_policy =
            opencl_context_->opencl_cache_reuse_policy();
        switch (opencl_cache_reuse_policy) {
          case OpenCLCacheReusePolicy::REUSE_NONE:
            cache_storage->Clear();
            break;
          case OpenCLCacheReusePolicy::REUSE_SAME_GPU:
            if (!IsCacheCompatible(gpu_type_,
                                   platform_version,
                                   cached_binary_platform_info)) {
              cache_storage->Clear();
            }
            break;
          default:
            cache_storage->Clear();
            break;
        }
      }
    }
  }

  if (cached_binary_platform_info != platform_info_) {
    auto precompiled_binary_storage = opencl_context_->opencl_binary_storage();
    if (precompiled_binary_storage == nullptr) {
      VLOG(1) << "There is no precompiled OpenCL binary in"
                 " all OpenCL binary paths.";
    } else {
      if (precompiled_binary_storage->Load() != 0 && !tuner->IsTuning()) {
        LOG(WARNING) << "Load OpenCL precompiled kernel file failed. "
                     << "Please make sure the storage directory exist "
                     << "and you have Write&Read permission";
      }

      auto platform_info_array =
          precompiled_binary_storage->Find(kOpenCLPlatformInfoKey);
      if (platform_info_array != nullptr) {
        precompiled_binary_platform_info_ =
            std::string(platform_info_array->begin(),
                        platform_info_array->end());
      }
    }
  }

  device_->getInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                   &device_global_mem_cache_size_);

  device_->getInfo(CL_DEVICE_MAX_COMPUTE_UNITS,
                   &device_compute_units_);
  const char *out_of_range_check = getenv("MACE_OUT_OF_RANGE_CHECK");
  if (out_of_range_check != nullptr && strlen(out_of_range_check) == 1
      && out_of_range_check[0] == '1') {
    this->out_of_range_check_ = true;
  } else {
    this->out_of_range_check_ = false;
  }

  is_opencl_avaliable_ = true;

  return MaceStatus::MACE_SUCCESS;
}

void OpenclExecutor::SetOpenclContext(
    std::shared_ptr<OpenclContext> opencl_context) {
  MACE_CHECK(opencl_context != nullptr);
  opencl_context_ = opencl_context;
}

OpenclExecutor::~OpenclExecutor() {
  if (command_queue_ != nullptr) {
    command_queue_->finish();
  }
  built_program_map_.clear();
  // We need to control the destruction order, which has dependencies
  command_queue_.reset();
  context_.reset();
  device_.reset();
}

void OpenclExecutor::InitGpuDeviceProperty(const cl::Device &device) {
  const std::string device_name = device.getInfo<CL_DEVICE_NAME>();
  VLOG(1) << "Using device: " << device_name;
  gpu_type_ = ParseGPUType(device_name);

  const std::string device_version = device.getInfo<CL_DEVICE_VERSION>();
  switch (gpu_type_) {
    case QUALCOMM_ADRENO:
      device_name_ = ParseAdrenoDeviceName(device_version);
      break;
    case MALI:
    case PowerVR:device_name_ = device_name;
      break;
    default:device_name_ = std::string();
      break;
  }
  opencl_version_ = ParseDeviceVersion(device_version);
  MACE_CHECK(opencl_version_ != OpenCLVersion::CL_VER_UNKNOWN);
}

bool OpenclExecutor::is_opencl_avaliable() {
  static const uint64_t kMinWorkGroupSize = 64;
  return is_opencl_avaliable_
      && GetDeviceMaxWorkGroupSize() >= kMinWorkGroupSize;
}

cl::Context &OpenclExecutor::context() { return *context_; }

cl::Device &OpenclExecutor::device() { return *device_; }

cl::CommandQueue &OpenclExecutor::command_queue() { return *command_queue_; }

std::shared_ptr<Tuner<uint32_t>> OpenclExecutor::tuner() {
  return opencl_context_->opencl_tuner();
}

uint64_t OpenclExecutor::device_global_mem_cache_size() const {
  return device_global_mem_cache_size_;
}

uint32_t OpenclExecutor::device_compute_units() const {
  return device_compute_units_;
}

inline MaceStatus ParseProgramNameByKey(const std::string &built_program_key,
                                        std::string *program_name) {
  size_t space_idx = built_program_key.find(' ');
  if (space_idx == std::string::npos) {
    *program_name = built_program_key;
  } else {
    *program_name = built_program_key.substr(0, space_idx);
  }
  return MaceStatus::MACE_SUCCESS;
}

inline MaceStatus GetProgramHashByName(const std::string &program_name,
                                       std::string *hash_str) {
  MACE_CHECK_NOTNULL(hash_str);
  const auto &kEncryptedProgramMap = mace::codegen::kEncryptedProgramMap;
  const auto &it_program = kEncryptedProgramMap.find(program_name);
  if (it_program == kEncryptedProgramMap.end()) {
    LOG(ERROR) << "Find program " << program_name << " failed.";
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
  *hash_str = it_program->second.hash_str_;
  return MaceStatus::MACE_SUCCESS;
}

bool OpenclExecutor::BuildProgramFromCache(
    const std::string &built_program_key,
    const std::string &build_options_str,
    cl::Program *program) {
  // Find from binary
  auto cache_storage = opencl_context_->opencl_cache_storage();
  if (cache_storage == nullptr) {
    return false;
  }
  std::string program_name;
  bool hash_match = false;
  std::string cached_program_hash;
  std::string current_program_hash;
  MaceStatus status = ParseProgramNameByKey(built_program_key, &program_name);
  if (status == MaceStatus::MACE_SUCCESS) {
    const std::vector<unsigned char> *hash_vec =
        cache_storage->Find(program_key_hash_prefix_ + built_program_key);
    if (hash_vec != nullptr) {
      cached_program_hash = std::string(hash_vec->begin(), hash_vec->end());
    }
    if (GetProgramHashByName(program_name, &current_program_hash) ==
        MaceStatus::MACE_SUCCESS) {
      // .cl file or header of .cl file is modified
      if (cached_program_hash.size() > 0 &&
          current_program_hash.size() > 0 &&
          current_program_hash == cached_program_hash) {
        hash_match = true;
      }
    }
  }
  if (!hash_match) {
    return false;
  }
  auto content = cache_storage->Find(built_program_key);
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

bool OpenclExecutor::BuildProgramFromPrecompiledBinary(
    const std::string &built_program_key,
    const std::string &build_options_str,
    cl::Program *program) {
  // Find from binary
  auto precompiled_binary_storage = opencl_context_->opencl_binary_storage();
  if (precompiled_binary_storage == nullptr) return false;
  if (precompiled_binary_platform_info_ != platform_info_) {
    VLOG(3) << "precompiled OpenCL binary version "
            << precompiled_binary_platform_info_
            << " is not same with current version";
    return false;
  }
  auto content = precompiled_binary_storage->Find(built_program_key);
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
  programs_need_store_.insert(built_program_key);
  VLOG(3) << "Program from precompiled binary: " << built_program_key;
  return true;
}

MaceStatus GetProgramSourceByName(const std::string &program_name,
                                  std::string *source) {
  MACE_CHECK_NOTNULL(source);
  std::stringstream source_stream;
  const auto &kEncryptedProgramMap = mace::codegen::kEncryptedProgramMap;
  const auto &it_program = kEncryptedProgramMap.find(program_name);
  if (it_program == kEncryptedProgramMap.end()) {
    LOG(ERROR) << "Find program " << program_name << " failed.";
    return MaceStatus::MACE_RUNTIME_ERROR;
  }

  const std::vector<std::string> &headers = it_program->second.headers_;
  for (const std::string &header : headers) {
    const auto &header_program = kEncryptedProgramMap.find(header);
    if (header_program == kEncryptedProgramMap.end()) {
      LOG(WARNING) << "Program header(" << header << ") is empty.";
      continue;
    }

    const auto &header_source = header_program->second.encrypted_code_;
    source_stream << ObfuscateString(
        std::string(header_source.begin(), header_source.end()));
  }

  const auto &it_source = it_program->second.encrypted_code_;
  source_stream << ObfuscateString(
      std::string(it_source.begin(), it_source.end()));
  *source = source_stream.str();

  return MaceStatus::MACE_SUCCESS;
}

bool OpenclExecutor::BuildProgramFromSource(
    const std::string &program_name,
    const std::string &built_program_key,
    const std::string &build_options_str,
    cl::Program *program) {
  std::string kernel_source;
  MaceStatus status = GetProgramSourceByName(program_name, &kernel_source);
  if (status == MaceStatus::MACE_SUCCESS && !kernel_source.empty()) {
    cl::Program::Sources sources;
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
      return false;
    }

    VLOG(3) << "Program from source: " << built_program_key;
    programs_need_store_.insert(built_program_key);
  }
  return true;
}

bool OpenclExecutor::BuildProgram(const std::string &program_name,
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
      ret = BuildProgramFromSource(program_name, built_program_key,
                                   build_options_str, program);
    }
  }
  return ret;
}

MaceStatus OpenclExecutor::BuildKernel(
    const std::string &program_name,
    const std::string &kernel_name,
    const std::set<std::string> &build_options,
    cl::Kernel *kernel) {
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
    bool ret = this->BuildProgram(program_name, built_program_key,
                                  build_options_str, &program);
    if (!ret) {
      return MaceStatus::MACE_OUT_OF_RESOURCES;
    }
    built_program_map_.emplace(built_program_key, program);
  }
  cl_int err;
  *kernel = cl::Kernel(program, kernel_name.c_str(), &err);
  MACE_CL_RET_STATUS(err);
  return MaceStatus::MACE_SUCCESS;
}

void OpenclExecutor::SaveBuiltCLProgram() {
  auto cache_storage = opencl_context_->opencl_cache_storage();
  if (programs_need_store_.empty() || cache_storage == nullptr) {
    return;
  }

  // update device name
  cache_storage->Insert(kOpenCLDeviceNameKey,
                        std::vector<unsigned char>(device_name_.begin(),
                                                   device_name_.end()));
  for (auto i = programs_need_store_.begin();
       i != programs_need_store_.end(); ++i) {
    auto &program_key = *i;
    MACE_CHECK(built_program_map_.count(program_key) > 0);
    auto &program = built_program_map_.at(program_key);

    auto content = GetBinaryFromProgram(program);
    cache_storage->Insert(program_key, content);

    std::string hash_str;
    std::string program_name;
    MaceStatus ret = ParseProgramNameByKey(program_key, &program_name);
    MACE_CHECK_SUCCESS(ret);
    ret = GetProgramHashByName(program_name, &hash_str);
    if (ret == MaceStatus::MACE_SUCCESS) {
      std::vector<unsigned char> hash_vec(hash_str.begin(), hash_str.end());
      // update program hash
      cache_storage->Insert(program_key_hash_prefix_ + program_key, hash_vec);
    } else {
      LOG(WARNING) << "Failed to get hash value of program " << program_name;
    }
  }

  // update platform info
  auto platform_info = std::vector<unsigned char>(platform_info_.begin(),
                                                  platform_info_.end());
  cache_storage->Insert(kOpenCLPlatformInfoKey, platform_info);

  if (cache_storage->Flush() != 0) {
    LOG(FATAL) << "Store OPENCL compiled kernel to file failed. "
               << "Please make sure the storage directory exist "
               << "and you have Write&Read permission";
  }

  programs_need_store_.clear();
}

void OpenclExecutor::GetCallStats(const cl::Event &event, CallStats *stats) {
  if (stats != nullptr) {
    stats->start_micros =
        event.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
    stats->end_micros =
        event.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;
  }
}

uint64_t OpenclExecutor::GetDeviceMaxWorkGroupSize() {
  uint64_t size = 0;
  cl_int err = device_->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    size = 0;
  }
  return size;
}

uint64_t OpenclExecutor::GetDeviceMaxMemAllocSize() {
  uint64_t size = 0;
  cl_int err = device_->getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    size = 0;
  }
  return size;
}

bool OpenclExecutor::IsImageSupport() {
  cl_bool res;
  cl_int err = device_->getInfo(CL_DEVICE_IMAGE_SUPPORT, &res);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    return false;
  }
  return res == CL_TRUE;
}
std::vector<uint64_t> OpenclExecutor::GetMaxImage2DSize() {
  size_t max_height, max_width;
  cl_int err = device_->getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &max_height);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    return {};
  }
  err = device_->getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &max_width);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    return {};
  }
  return {max_width, max_height};
}

uint64_t OpenclExecutor::GetKernelMaxWorkGroupSize(const cl::Kernel &kernel) {
  uint64_t size = 0;
  cl_int err = kernel.getWorkGroupInfo(*device_, CL_KERNEL_WORK_GROUP_SIZE,
                                       &size);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    size = 0;
  }
  return size;
}

uint64_t OpenclExecutor::GetKernelWaveSize(const cl::Kernel &kernel) {
  uint64_t size = 0;
  cl_int err = kernel.getWorkGroupInfo(*device_, CL_KERNEL_WAVE_SIZE_QCOM,
                                       &size);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "error: " << OpenCLErrorToString(err);
    size = 0;
  }
  return size;
}

bool OpenclExecutor::IsNonUniformWorkgroupsSupported() const {
  return (gpu_type_ == GPUType::QUALCOMM_ADRENO &&
      opencl_version_ >= OpenCLVersion::CL_VER_2_0);
}

GPUType OpenclExecutor::gpu_type() const {
  return gpu_type_;
}

IONType OpenclExecutor::FindCurDeviceIonType() {
  constexpr const char *kQualcommIONStr = "cl_qcom_ion_host_ptr";
  std::shared_ptr<cl::Device> device = FindGpuDevice(FindGpuPlatform());
  const auto device_extensions = device->getInfo<CL_DEVICE_EXTENSIONS>();
  if (device_extensions.find(kQualcommIONStr) != std::string::npos) {
    return IONType::QUALCOMM_ION;
  } else {
    return IONType::NONE_ION;
  }
}

IONType OpenclExecutor::ion_type() const {
  return IONType::NONE_ION;
}

const std::string OpenclExecutor::platform_info() const {
  return platform_info_;
}

OpenCLVersion OpenclExecutor::ParseDeviceVersion(
    const std::string &device_version) {
  // OpenCL Device version string format:
  // OpenCL<space><major_version.minor_version><space>
  // <vendor-specific information>
  auto words = Split(device_version, ' ');
  if (words[1] == "2.1") {
    return OpenCLVersion::CL_VER_2_1;
  } else if (words[1] == "2.0") {
    return OpenCLVersion::CL_VER_2_0;
  } else if (words[1] == "1.2") {
    return OpenCLVersion::CL_VER_1_2;
  } else if (words[1] == "1.1") {
    return OpenCLVersion::CL_VER_1_1;
  } else if (words[1] == "1.0") {
    return OpenCLVersion::CL_VER_1_0;
  } else {
    LOG(ERROR) << "Do not support OpenCL version: " << words[1];
    return OpenCLVersion::CL_VER_UNKNOWN;
  }
}

std::string OpenclExecutor::ParseAdrenoDeviceName(
    const std::string &device_version) {
  // Adreno OpenCL Device version string format:
  // OpenCL<space><major_version.minor_version><space>
  // <device name>
  int space_count = 0;
  int str_idx = 0;
  int len = static_cast<int>(device_version.size());
  for (; str_idx < len && space_count != 2; ++str_idx) {
    if (device_version[str_idx] == ' ') {
      ++space_count;
    }
  }
  if (space_count != 2) {
    return std::string();
  }
  return device_version.substr(str_idx, len - str_idx);
}

bool OpenclExecutor::IsOutOfRangeCheckEnabled() const {
  return out_of_range_check_;
}

bool OpenclExecutor::is_profiling_enabled() const {
  return is_profiling_enabled_;
}

}  // namespace mace
