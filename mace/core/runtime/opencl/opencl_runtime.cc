//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>

#include <dirent.h>

#include "mace/core/logging.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {
namespace {

bool ReadSourceFile(const std::string &filename, std::string *content) {
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
    *content += "\n";
  }
  ifs.close();
  return true;
}

bool BuildProgram(OpenCLRuntime *runtime,
                  const std::string &path,
                  cl::Program *program) {
  MACE_CHECK_NOTNULL(program);

  auto closer = [](DIR *d) {
    if (d != nullptr) closedir(d);
  };
  std::unique_ptr<DIR, decltype(closer)> dir(opendir(path.c_str()), closer);
  MACE_CHECK_NOTNULL(dir.get());

  const std::string kSourceSuffix = ".cl";
  cl::Program::Sources sources;
  errno = 0;
  dirent *entry = readdir(dir.get());
  MACE_CHECK(errno == 0);
  while (entry != nullptr) {
    if (entry->d_type == DT_REG) {
      std::string d_name(entry->d_name);
      if (d_name.size() > kSourceSuffix.size() &&
          d_name.compare(d_name.size() - kSourceSuffix.size(),
                         kSourceSuffix.size(), kSourceSuffix) == 0) {
        std::string filename = path + d_name;
        std::string kernel_source;
        MACE_CHECK(ReadSourceFile(filename, &kernel_source));
        sources.push_back({kernel_source.c_str(), kernel_source.length()});
      }
    }
    entry = readdir(dir.get());
    MACE_CHECK(errno == 0);
  };

  *program = cl::Program(runtime->context(), sources);
  std::string build_options = "-Werror -cl-mad-enable -cl-fast-relaxed-math -I" + path;
  // TODO(heliangliang) -cl-unsafe-math-optimizations -cl-fast-relaxed-math
  cl_int ret = program->build({runtime->device()}, build_options.c_str());
  if (ret != CL_SUCCESS) {
    if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(runtime->device()) ==
        CL_BUILD_ERROR) {
      std::string build_log =
          program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(runtime->device());
      LOG(INFO) << "Program build log: " << build_log;
    }
    LOG(FATAL) << "Build program failed: " << ret;
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

cl::Program &OpenCLRuntime::program() {
  // TODO(heliangliang) Support binary format
  static const char *kernel_path = getenv("MACE_KERNEL_PATH");
  std::string path(kernel_path == nullptr ? "" : kernel_path);

  std::call_once(build_flag_, [this, &path]() {
    MACE_CHECK(BuildProgram(this, path, &program_));
  });

  return program_;
}

uint32_t OpenCLRuntime::GetDeviceMaxWorkGroupSize() {
  unsigned long long size = 0;
  device_.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);
  return static_cast<uint32_t>(size);
}

uint32_t OpenCLRuntime::GetKernelMaxWorkGroupSize(const cl::Kernel& kernel) {
  unsigned long long size = 0;
  kernel.getWorkGroupInfo(device_, CL_KERNEL_WORK_GROUP_SIZE, &size);
  return static_cast<uint32_t>(size);
}

}  // namespace mace
