//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <cstdlib>
#include <fstream>
#include <mutex>

#include "mace/core/logging.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/runtime/opencl/opencl_wrapper.h"

namespace mace {
namespace {

bool ReadSourceFile(const char *filename, std::string *content) {
  MACE_CHECK_NOTNULL(filename);
  MACE_CHECK_NOTNULL(content);
  *content = "";
  std::ifstream ifs(filename, std::ifstream::in);
  if (!ifs.is_open()) {
    LOG(ERROR) << "Failed to open file " << filename;
    return false;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    *content += line;
  }
  ifs.close();
  return true;
}

bool BuildProgram(OpenCLRuntime *runtime,
                  const char *filename,
                  cl::Program *program) {
  MACE_CHECK_NOTNULL(filename);
  MACE_CHECK_NOTNULL(program);

  std::string kernel_code;
  if (!ReadSourceFile(filename, &kernel_code)) {
    LOG(ERROR) << "Failed to read kernel source " << filename;
    return false;
  }

  cl::Program::Sources sources;
  sources.push_back({kernel_code.c_str(), kernel_code.length()});

  *program = cl::Program(runtime->context(), sources);
  if (program->build({runtime->device()}) != CL_SUCCESS) {
    LOG(INFO) << "Error building: "
              << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(runtime->device());
    return false;
  }
  return true;
}

}  // namespace

OpenCLRuntime *OpenCLRuntime::Get() {
  static std::once_flag init_once;
  static OpenCLRuntime *instance = nullptr;
  std::call_once(init_once, []() {
    if (!mace::OpenCLLibrary::Supported()) {
      LOG(ERROR) << "OpenCL not supported";
      return;
    }

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
      LOG(ERROR) << "No OpenCL platforms found";
      return;
    }
    cl::Platform default_platform = all_platforms[0];
    VLOG(1) << "Using platform: "
            << default_platform.getInfo<CL_PLATFORM_NAME>() << ", "
            << default_platform.getInfo<CL_PLATFORM_PROFILE>() << ", "
            << default_platform.getInfo<CL_PLATFORM_VERSION>();

    // get default device (CPUs, GPUs) of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
      LOG(ERROR) << "No OpenCL devices found";
      return;
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
      LOG(ERROR) << "No GPU device found";
      return;
    }

    // a context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    cl::Context context({gpu_device});
    cl::CommandQueue command_queue(context, gpu_device);
    instance = new OpenCLRuntime(context, gpu_device, command_queue);
  });

  return instance;
}

OpenCLRuntime::OpenCLRuntime(cl::Context context,
                             cl::Device device,
                             cl::CommandQueue command_queue)
    : context_(context), device_(device), command_queue_(command_queue) {}

OpenCLRuntime::~OpenCLRuntime() {}

cl::Context &OpenCLRuntime::context() { return context_; }

cl::Device &OpenCLRuntime::device() { return device_; }

cl::CommandQueue &OpenCLRuntime::command_queue() { return command_queue_; }

cl::Program OpenCLRuntime::GetProgram(const std::string &name) {
  static const char *kernel_source_path = getenv("MACE_KERNEL_SOURCE_PATH");
  std::string filename = name;
  if (kernel_source_path != nullptr) {
    filename = kernel_source_path + name;
  }

  std::lock_guard<std::mutex> lock(program_lock_);
  // TODO (heliangliang) Support binary format
  auto iter = programs_.find(name);
  if (iter != programs_.end()) {
    return iter->second;
  } else {
    cl::Program program;
    MACE_CHECK(BuildProgram(this, filename.c_str(), &program));
    programs_.emplace(name, program);
    return program;
  }
}

}  // namespace mace
