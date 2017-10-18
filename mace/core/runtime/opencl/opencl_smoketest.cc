//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/logging.h"
#include "mace/core/operator.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/runtime/opencl/opencl_wrapper.h"

int main() {
  using namespace mace;
  auto runtime = mace::OpenCLRuntime::Get();

  mace::Tensor ta(GetDeviceAllocator(DeviceType::OPENCL), DataType::DT_INT32);
  mace::Tensor tb(GetDeviceAllocator(DeviceType::OPENCL), DataType::DT_INT32);
  mace::Tensor tc(GetDeviceAllocator(DeviceType::OPENCL), DataType::DT_INT32);
  mace::Tensor tstep(GetDeviceAllocator(DeviceType::OPENCL),
                     DataType::DT_INT32);

  int n = 1000;
  std::vector<index_t> shape = {n};
  ta.Resize(shape);
  tb.Resize(shape);
  tc.Resize(shape);
  tstep.Resize({1});

  int step_size = 10;
  int global_size = n / step_size;
  {
    mace::Tensor::MappingGuard ta_mapper(&ta);
    mace::Tensor::MappingGuard tb_mapper(&tb);
    mace::Tensor::MappingGuard tstep_mapper(&tstep);
    int32_t *a = ta.mutable_data<int32_t>();
    int32_t *b = tb.mutable_data<int32_t>();
    int32_t *step = tstep.mutable_data<int32_t>();
    for (int i = 0; i < n; i++) {
      a[i] = i;
      b[i] = 2 * i;
    }
    step[0] = step_size;
  }

  auto program = runtime->program();

  auto simple_add =
      cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(
          program, "simple_add");
  cl_int error;
  simple_add(cl::EnqueueArgs(runtime->command_queue(), cl::NDRange(global_size),
                             cl::NullRange),
             *(static_cast<cl::Buffer *>(ta.buffer())),
             *(static_cast<cl::Buffer *>(tb.buffer())),
             *(static_cast<cl::Buffer *>(tc.buffer())),
             *(static_cast<cl::Buffer *>(tstep.buffer())), error);
  if (error != 0) {
    LOG(ERROR) << "Failed to execute kernel " << error;
  }

  {
    mace::Tensor::MappingGuard ta_mapper(&ta);
    mace::Tensor::MappingGuard tb_mapper(&tb);
    mace::Tensor::MappingGuard tc_mapper(&tc);

    int32_t *a = ta.mutable_data<int32_t>();
    int32_t *b = tb.mutable_data<int32_t>();
    int32_t *c = tc.mutable_data<int32_t>();
    bool correct = true;
    for (int i = 0; i < n; i++) {
      if (c[i] != a[i] + b[i]) correct = false;
    }
    LOG(INFO) << "OpenCL test result: " << (correct ? "correct" : "incorrect");
  }

  return 0;
}
