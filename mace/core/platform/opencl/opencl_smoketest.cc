//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include "mace/core/logging.h"
#include "mace/core/platform/opencl/cl2.hpp"
#include "mace/core/platform/opencl/opencl_wrapper.h"

int main() {
  LOG(INFO) << "OpenCL support: " << mace::OpenCLSupported();
  if (!mace::OpenCLSupported()) return 1;
  LOG(INFO) << "Start OpenCL test";

  // get all platforms (drivers)
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  if (all_platforms.size() == 0) {
    LOG(INFO) << " No OpenCL platforms found";
    return 1;
  }
  LOG(INFO) << "Platform sizes: " << all_platforms.size();
  cl::Platform default_platform = all_platforms[0];
  LOG(INFO) << "Using platform: "
            << default_platform.getInfo<CL_PLATFORM_NAME>() << ", "
            << default_platform.getInfo<CL_PLATFORM_PROFILE>() << ", "
            << default_platform.getInfo<CL_PLATFORM_VERSION>();

  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if (all_devices.size() == 0) {
    LOG(INFO) << "No OpenCL devices found";
    return 1;
  }

  // Use the last device
  cl::Device default_device = *all_devices.rbegin();
  LOG(INFO) << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>()
            << ", " << default_device.getInfo<CL_DEVICE_TYPE>();

  // a context is like a "runtime link" to the device and platform;
  // i.e. communication is possible
  cl::Context context({default_device});

  // create the program that we want to execute on the device
  cl::Program::Sources sources;

  // calculates for each element; C = A + B
  std::string kernel_code =
      "   void kernel simple_add(global const int* A, global const int* B, "
      "global int* C, "
      "                          global const int* N) {"
      "       int ID, Nthreads, n, ratio, start, stop;"
      ""
      "       ID = get_global_id(0);"
      "       Nthreads = get_global_size(0);"
      "       n = N[0];"
      ""
      "       ratio = (n / Nthreads);"  // number of elements for each thread
      "       start = ratio * ID;"
      "       stop  = ratio * (ID + 1);"
      ""
      "       for (int i=start; i<stop; i++)"
      "           C[i] = A[i] + B[i];"
      "   }";
  sources.push_back({kernel_code.c_str(), kernel_code.length()});

  cl::Program program(context, sources);
  if (program.build({default_device}) != CL_SUCCESS) {
    LOG(INFO) << "Error building: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device);
    return 1;
  }

  // apparently OpenCL only likes arrays ...
  // N holds the number of elements in the vectors we want to add
  int N[1] = {1000};
  int n = N[0];

  // create buffers on device (allocate space on GPU)
  cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
  cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
  cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);
  cl::Buffer buffer_N(context, CL_MEM_READ_ONLY, sizeof(int));

  // create things on here (CPU)
  int A[n], B[n];
  for (int i = 0; i < n; i++) {
    A[i] = i;
    B[i] = 2 * i;
  }
  // create a queue (a queue of commands that the GPU will execute)
  cl::CommandQueue queue(context, default_device);

  // push write commands to queue
  queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * n, A);
  queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * n, B);
  queue.enqueueWriteBuffer(buffer_N, CL_TRUE, 0, sizeof(int), N);

  auto simple_add =
      cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(
          program, "simple_add");
  cl_int error;
  simple_add(cl::EnqueueArgs(queue, cl::NDRange(100), cl::NDRange(10)),
             buffer_A, buffer_B, buffer_C, buffer_N, error);
  if (error != 0) {
    LOG(ERROR) << "Failed to execute kernel " << error;
  }

  int C[n];
  // read result from GPU to here
  queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * n, C);

  bool correct = true;
  for (int i = 0; i < n; i++) {
    if (C[i] != A[i] + B[i]) correct = false;
  }
  LOG(INFO) << "OpenCL test result: " << (correct ? "correct" : "incorrect");

  return 0;
}
