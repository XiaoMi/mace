//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>

#include "mace/core/logging.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

#include <CL/opencl.h>

namespace mace {
namespace {

bool ReadFile(const std::string &filename, std::string &content, bool binary) {
  content = "";

  std::ios_base::openmode mode = std::ios::in;
  if (binary)
    mode |= std::ios::binary;

  std::ifstream ifs(filename, mode);

  if (!ifs.is_open()) {
    LOG(ERROR) << "Failed to open file " << filename;
    return false;
  }

  ifs.seekg(0, std::ios::end);
  content.reserve(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  content.assign(std::istreambuf_iterator<char>(ifs),
                 std::istreambuf_iterator<char>());

  ifs.close();
  return true;
}

bool WriteFile(const std::string &filename, std::string &content, bool binary) {
  std::ios_base::openmode mode = std::ios::out;
  if (binary)
    mode |= std::ios::binary;
  std::ofstream ofs(filename, mode);

  ofs.write(content.c_str(), content.size());
  ofs.close();

  return true;
}

}  // namespace

bool OpenCLRuntime::enable_profiling_ = false;
std::unique_ptr<cl::Event> OpenCLRuntime::profiling_ev_ = NULL;

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

    cl_command_queue_properties properties = 0;
#ifdef __ENABLE_PROFILING
    enable_profiling_ = true;
    profiling_ev_.reset(new cl::Event());
    properties = CL_QUEUE_PROFILING_ENABLE;
#endif

    // a context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    cl::Context context({gpu_device});
    cl::CommandQueue command_queue(context, gpu_device, properties);
    instance = new OpenCLRuntime(context, gpu_device, command_queue);

  });

  return instance;
}

void OpenCLRuntime::EnableProfiling() {
  enable_profiling_ = true;
}

cl::Event* OpenCLRuntime::GetDefaultEvent() {
  return profiling_ev_.get();
}

cl_ulong OpenCLRuntime::GetEventProfilingStartInfo() {
  MACE_CHECK(profiling_ev_, "is NULL, should enable profiling first.");
  return profiling_ev_->getProfilingInfo<CL_PROFILING_COMMAND_START>();
}

cl_ulong OpenCLRuntime::GetEventProfilingEndInfo() {
  MACE_CHECK(profiling_ev_, "is NULL, should enable profiling first.");
  return profiling_ev_->getProfilingInfo<CL_PROFILING_COMMAND_END>();
}

OpenCLRuntime::OpenCLRuntime(cl::Context context,
                             cl::Device device,
                             cl::CommandQueue command_queue)
    : context_(context), device_(device), command_queue_(command_queue) {
  const char *kernel_path = getenv("MACE_KERNEL_PATH");
  kernel_path_ = std::string(kernel_path == nullptr ? "" : kernel_path) + "/";
}

OpenCLRuntime::~OpenCLRuntime() {}

cl::Context &OpenCLRuntime::context() { return context_; }

cl::Device &OpenCLRuntime::device() { return device_; }

cl::CommandQueue &OpenCLRuntime::command_queue() { return command_queue_; }

cl::Program &OpenCLRuntime::program() {
  // TODO(liuqi) : useless, leave it for old code.
  return program_;
}

// TODO(heliangliang) Support binary format
const std::map<std::string, std::string>
    OpenCLRuntime::program_map_ = {
  {"addn", "addn.cl"},
  {"batch_norm", "batch_norm.cl"},
  {"bias_add", "bias_add.cl"},
  {"buffer_to_image", "buffer_to_image.cl"},
  {"conv_2d", "conv_2d.cl"},
  {"conv_2d_1x1", "conv_2d_1x1.cl"},
  {"conv_2d_3x3", "conv_2d_3x3.cl"},
  {"depthwise_conv_3x3", "depthwise_conv_3x3.cl"},
  {"pooling", "pooling.cl"},
  {"relu", "relu.cl"},
  {"concat", "concat.cl"},
  {"resize_bilinear", "resize_bilinear.cl"},
  {"space_to_batch", "space_to_batch.cl"},
};

const std::map<std::string, std::string>
    OpenCLRuntime::binary_map_ = {
  {"addn -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DINPUT_NUM=2", "addn_f_2.bin"},
  {"addn -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DINPUT_NUM=3", "addn_f_3.bin"},
  {"addn -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DINPUT_NUM=4", "addn_f_4.bin"},
  {"batch_norm -DCMD_DATA_TYPE=f -DDATA_TYPE=float", "batch_norm_f.bin"},
  {"bias_add -DCMD_DATA_TYPE=f -DDATA_TYPE=float", "bias_add_f.bin"},
  {"buffer_to_image -DCMD_DATA_TYPE=f -DDATA_TYPE=float", "buffer_to_image_f.bin"},
  {"buffer_to_image -DCMD_DATA_TYPE=h -DDATA_TYPE=half", "buffer_to_image_h.bin"},
  {"concat -DCMD_DATA_TYPE=f -DDATA_TYPE=float", "concat_f.bin"},
  {"concat -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DDIVISIBLE_FOUR", "concat_f_div4.bin"},
  {"concat -DCMD_DATA_TYPE=h -DDATA_TYPE=half -DDIVISIBLE_FOUR", "concat_h_div4.bin"},
  {"conv_2d -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DFUSED_RELU -DSTRIDE=1", "conv_2d_bias_f_fusedrelu_1.bin"},
  {"conv_2d -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DFUSED_RELU -DSTRIDE=2", "conv_2d_bias_f_fusedrelu_2.bin"},
  {"conv_2d -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DSTRIDE=1", "conv_2d_bias_f_1.bin"},
  {"conv_2d -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DSTRIDE=2", "conv_2d_bias_f_2.bin"},
  {"conv_2d_1x1 -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DFUSED_RELU -DSTRIDE=1", "conv_2d_1x1_bias_f_fusedrelu_1.bin"},
  {"conv_2d_1x1 -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DFUSED_RELU -DSTRIDE=2", "conv_2d_1x1_bias_f_fusedrelu_2.bin"},
  {"conv_2d_1x1 -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DSTRIDE=1", "conv_2d_1x1_bias_f_1.bin"},
  {"conv_2d_1x1 -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DSTRIDE=2", "conv_2d_1x1_bias_f_2.bin"},
  {"conv_2d_3x3  -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DSTRIDE=1", "conv_2d_3x3_f_1.bin"},
  {"conv_2d_3x3 -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DFUSED_RELU -DSTRIDE=1", "conv_2d_3x3_bias_f_fusedrelu_1.bin"},
  {"conv_2d_3x3 -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DFUSED_RELU -DSTRIDE=2", "conv_2d_3x3_bias_f_fusedrelu_2.bin"},
  {"conv_2d_3x3 -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DSTRIDE=1", "conv_2d_3x3_bias_f_1.bin"},
  {"conv_2d_3x3 -DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DSTRIDE=2", "conv_2d_3x3_bias_f_2.bin"},
  {"depthwise_conv_3x3  -DBIAS -DDATA_TYPE=float", "depthwise_conv_3x3_bias_f.bin"},
  {"depthwise_conv_3x3 -DBIAS -DDATA_TYPE=float -DSTRIDE_1", "depthwise_conv_3x3_bias_f_1.bin"},
  {"pooling  -DCMD_DATA_TYPE=f -DDATA_TYPE=float", "pooling_f.bin"},
  {"pooling -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DPOOL_AVG", "pooling_f_avg.bin"},
  {"pooling -DCMD_DATA_TYPE=h -DDATA_TYPE=half -DFP16", "pooling_h_fp16.bin"},
  {"relu -DCMD_DATA_TYPE=f -DDATA_TYPE=float", "relu_f.bin"},
  {"resize_bilinear -DCMD_DATA_TYPE=f -DDATA_TYPE=float", "resize_bilinear_f.bin"},
  {"space_to_batch -DDATA_TYPE=float", "space_to_batch_f.bin"},
};

void OpenCLRuntime::BuildProgram(const std::string &program_file_name,
                                 const std::string &binary_file_name,
                                 const std::string &build_options,
                                 cl::Program *program) {
  MACE_CHECK_NOTNULL(program);

  std::string source_filename = kernel_path_ + program_file_name;
  std::string binary_filename = kernel_path_ + binary_file_name;

  // Create program
  if (std::ifstream(binary_filename).is_open()) {
    VLOG(1) << "Create program with binary: " << binary_filename;
    std::string kernel_binary;
    MACE_CHECK(ReadFile(binary_filename, kernel_binary, true));

    std::vector<unsigned char> binaries(
        kernel_binary.begin(), kernel_binary.end());

    *program = cl::Program(this->context(), {device()}, {binaries});

  } else if (std::ifstream(source_filename).is_open()) {
    VLOG(1) << "Create program with source: " << source_filename;
    std::string kernel_source;
    MACE_CHECK(ReadFile(source_filename, kernel_source, false));

    cl::Program::Sources sources;
    sources.push_back({kernel_source.c_str(), kernel_source.length()});

    *program = cl::Program(this->context(), sources);

  } else {
    LOG(ERROR) << "Failed to open kernel file " << binary_filename << " and "
               << source_filename;
  }

  // Build program
  std::string build_options_str = build_options +
      " -Werror -cl-mad-enable -cl-fast-relaxed-math -I" + kernel_path_;
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

  // Write binary if necessary
  if (!std::ifstream(binary_filename).is_open()) {
    size_t deviceListSize = 1;
    size_t *programBinarySizes = new size_t[deviceListSize];
    clGetProgramInfo((*program)(),
                     CL_PROGRAM_BINARY_SIZES,
                     sizeof(size_t) * deviceListSize,
                     programBinarySizes,
                     NULL);
    unsigned char **programBinaries = new unsigned char *[deviceListSize];
    for(cl_uint i = 0; i < deviceListSize; ++i)
      programBinaries[i] = new unsigned char[programBinarySizes[i]];

    clGetProgramInfo((*program)(),
                     CL_PROGRAM_BINARIES,
                     sizeof(unsigned char *) * deviceListSize,
                     programBinaries,
                     NULL);
    std::string content(reinterpret_cast<char const*>(programBinaries[0]),
                        programBinarySizes[0]);

    WriteFile(binary_filename, content, true);
  }

}

cl::Kernel OpenCLRuntime::BuildKernel(const std::string &program_name,
                                      const std::string &kernel_name,
                                      const std::set<std::string> &build_options) {
  auto kernel_program_it = program_map_.find(program_name);
  if (kernel_program_it == program_map_.end()) {
    MACE_CHECK(false, program_name, " opencl kernel doesn't exist.");
  }

  std::string program_file_name = kernel_program_it->second;
  std::string build_options_str;
  for(auto &option : build_options) {
    build_options_str += " " + option;
  }
  std::string built_program_key = program_name + build_options_str;

  std::lock_guard<std::mutex> lock(program_build_mutex_);
  auto built_program_it = built_program_map_.find(built_program_key);
  cl::Program program;
  if (built_program_it != built_program_map_.end()) {
    program = built_program_it->second;
  } else {
    std::string binary_file_name = "";
    auto binary_it = binary_map_.find(built_program_key);
    if (binary_it != binary_map_.end()) {
      binary_file_name = binary_it->second;
    }
    this->BuildProgram(program_file_name,
                       binary_file_name,
                       build_options_str,
                       &program);
    built_program_map_.emplace(built_program_key, program);
  }
  return cl::Kernel(program, kernel_name.c_str());
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
