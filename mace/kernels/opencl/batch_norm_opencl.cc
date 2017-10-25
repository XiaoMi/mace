//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/batch_norm.h"
#include "mace/core/runtime/opencl/cl2.hpp"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {
namespace kernels {

template <>
void BatchNormFunctor<DeviceType::OPENCL, float>::operator()(
    const Tensor *input,
    const Tensor *scale,
    const Tensor *offset,
    const Tensor *mean,
    const Tensor *var,
    const Tensor *epsilon,
    Tensor *output) {
  const index_t n = input->dim(0);
  const index_t channel = input->dim(1);
  const index_t sample_size = input->dim(2) * input->dim(3);

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  auto _kernel = cl::Kernel(program, "batch_norm");
  _kernel.setArg(0, *(static_cast<const cl::Buffer *>(input->buffer())));
  _kernel.setArg(1, *(static_cast<cl::Buffer *>(scale->buffer())));
  _kernel.setArg(2, *(static_cast<cl::Buffer *>(offset->buffer())));
  _kernel.setArg(3, *(static_cast<cl::Buffer *>(mean->buffer())));
  _kernel.setArg(4, *(static_cast<cl::Buffer *>(var->buffer())));
  _kernel.setArg(5, *(static_cast<cl::Buffer *>(epsilon->buffer())));
  _kernel.setArg(6, static_cast<int>(sample_size));
  _kernel.setArg(7, *(static_cast<cl::Buffer *>(output->buffer())));
  _kernel.setArg(8, 32u, nullptr);
  _kernel.setArg(9, 32u, nullptr);
  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      _kernel, cl::NullRange,
      cl::NDRange(n, channel, sample_size),
      cl::NDRange(1, 1, 128));
  MACE_CHECK(error == CL_SUCCESS);
}

}  // namespace kernels
}  //  namespace mace