//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>

#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/utils/logging.h"
#include "mace/utils/tuner.h"

#include <CL/opencl.h>

namespace mace {
namespace {

bool WriteFile(const std::string &filename,
               bool binary,
               const std::vector<unsigned char> &content) {
  std::ios_base::openmode mode = std::ios::out;
  if (binary) {
    mode |= std::ios::binary;
  }
  std::ofstream ofs(filename, mode);

  ofs.write(reinterpret_cast<const char *>(&content[0]),
            content.size() * sizeof(char));
  ofs.close();
  if (ofs.fail()) {
    LOG(ERROR) << "Failed to write to file " << filename;
    return false;
  }

  return true;
}

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

OpenCLRuntime *OpenCLRuntime::Global() {
  static OpenCLRuntime instance;
  return &instance;
}

OpenCLRuntime::OpenCLRuntime() {
  LoadOpenCLLibrary();

  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
    LOG(FATAL) << "No OpenCL platforms found";
  }
  cl::Platform default_platform = all_platforms[0];
  VLOG(1) << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>()
          << ", " << default_platform.getInfo<CL_PLATFORM_PROFILE>() << ", "
          << default_platform.getInfo<CL_PLATFORM_VERSION>();

  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if (all_devices.size() == 0) {
    LOG(FATAL) << "No OpenCL devices found";
  }

  bool gpu_detected = false;
  cl::Device gpu_device;
  for (auto device : all_devices) {
    if (device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
      gpu_device = device;
      gpu_detected = true;
      VLOG(1) << "Using device: " << device.getInfo<CL_DEVICE_NAME>();
      break;
    }
  }
  if (!gpu_detected) {
    LOG(FATAL) << "No GPU device found";
  }

  cl_command_queue_properties properties = 0;

#ifdef MACE_OPENCL_PROFILING
  properties |= CL_QUEUE_PROFILING_ENABLE;
#endif

  // a context is like a "runtime link" to the device and platform;
  // i.e. communication is possible
  cl::Context context({gpu_device});
  cl::CommandQueue command_queue(context, gpu_device, properties);

  const char *kernel_path = getenv("MACE_KERNEL_PATH");
  this->kernel_path_ = std::string(kernel_path == nullptr ? "" : kernel_path) + "/";

  this->device_ = new cl::Device(gpu_device);
  this->context_ = new cl::Context(context);
  this->command_queue_ = new cl::CommandQueue(command_queue);
}

OpenCLRuntime::~OpenCLRuntime() {
  built_program_map_.clear();
  delete command_queue_;
  delete context_;
  delete device_;
  UnloadOpenCLLibrary();
}

cl::Context &OpenCLRuntime::context() { return *context_; }

cl::Device &OpenCLRuntime::device() { return *device_; }

cl::CommandQueue &OpenCLRuntime::command_queue() { return *command_queue_; }

std::string OpenCLRuntime::GenerateCLBinaryFilenamePrefix(
    const std::string &filename_msg) {
  std::string filename_prefix = filename_msg;
  for (auto it = filename_prefix.begin(); it != filename_prefix.end(); ++it) {
    if (*it == ' ' || *it == '-' || *it == '=') {
      *it = '_';
    }
  }
  return filename_prefix;
}

extern bool GetSourceOrBinaryProgram(const std::string &program_name,
                                     const std::string &binary_file_name_prefix,
                                     cl::Context &context,
                                     cl::Device &device,
                                     cl::Program *program,
                                     bool *is_opencl_binary);

void OpenCLRuntime::BuildProgram(const std::string &program_name,
                                 const std::string &binary_file_name_prefix,
                                 const std::string &build_options,
                                 cl::Program *program) {
  MACE_CHECK_NOTNULL(program);

  bool is_opencl_binary = false;
  std::vector<unsigned char> program_vec;
  const bool found = GetSourceOrBinaryProgram(program_name,
                                              binary_file_name_prefix,
                                              context(),
                                              device(),
                                              program,
                                              &is_opencl_binary);
  MACE_CHECK(found, "Program not found source: ", program_name, ", or binary: ",
             binary_file_name_prefix);

  // Build program
  std::string build_options_str =
      build_options + " -Werror -cl-mad-enable -cl-fast-relaxed-math";
  // TODO(heliangliang) -cl-unsafe-math-optimizations -cl-fast-relaxed-math
  cl_int ret = program->build({device()}, build_options_str.c_str());
  if (ret != CL_SUCCESS) {
    if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device()) ==
        CL_BUILD_ERROR) {
      std::string build_log =
          program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device());
      LOG(INFO) << "Program build log: " << build_log;
    }
    LOG(FATAL) << "Build program failed: " << ret;
  }

  if (!is_opencl_binary) {
    // Write binary if necessary
    std::string binary_filename = kernel_path_ + binary_file_name_prefix + ".bin";
    size_t device_list_size = 1;
    std::unique_ptr<size_t[]> program_binary_sizes(
        new size_t[device_list_size]);
    cl_int err = clGetProgramInfo((*program)(), CL_PROGRAM_BINARY_SIZES,
                                  sizeof(size_t) * device_list_size,
                                  program_binary_sizes.get(), nullptr);
    MACE_CHECK(err == CL_SUCCESS) << "Error code: " << err;
    std::unique_ptr<std::unique_ptr<unsigned char[]>[]> program_binaries(
        new std::unique_ptr<unsigned char[]>[device_list_size]);
    for (cl_uint i = 0; i < device_list_size; ++i) {
      program_binaries[i] = std::unique_ptr<unsigned char[]>(
          new unsigned char[program_binary_sizes[i]]);
    }

    err = clGetProgramInfo((*program)(), CL_PROGRAM_BINARIES,
                           sizeof(unsigned char *) * device_list_size,
                           program_binaries.get(), nullptr);
    MACE_CHECK(err == CL_SUCCESS) << "Error code: " << err;
    std::vector<unsigned char> content(
        reinterpret_cast<unsigned char const *>(program_binaries[0].get()),
        reinterpret_cast<unsigned char const *>(program_binaries[0].get()) +
            program_binary_sizes[0]);

    MACE_CHECK(WriteFile(binary_filename, true, content));
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
    std::string binary_file_name_prefix =
        GenerateCLBinaryFilenamePrefix(built_program_key);
    this->BuildProgram(program_name, binary_file_name_prefix,
                       build_options_str, &program);
    built_program_map_.emplace(built_program_key, program);
  }
  return cl::Kernel(program, kernel_name.c_str());
}

void OpenCLRuntime::GetCallStats(const cl::Event &event, CallStats *stats) {
  if (stats != nullptr) {
    stats->start_micros =
      event.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
    stats->end_micros =
      event.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;
  }
}

uint32_t OpenCLRuntime::GetDeviceMaxWorkGroupSize() {
  unsigned long long size = 0;
  device_->getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
  return static_cast<uint32_t>(size);
}

uint32_t OpenCLRuntime::GetKernelMaxWorkGroupSize(const cl::Kernel &kernel) {
  unsigned long long size = 0;
  kernel.getWorkGroupInfo(*device_, CL_KERNEL_WORK_GROUP_SIZE, &size);
  return static_cast<uint32_t>(size);
}

}  // namespace mace
