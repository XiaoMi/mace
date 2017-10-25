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
  auto batch_norm_kernel =
      cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,
                        cl::Buffer, cl::Buffer, cl::Buffer,
                        int, int, cl::Buffer>(program, "batch_norm");
  cl_int error;
  auto res_event = batch_norm_kernel(cl::EnqueueArgs(runtime->command_queue(),
                              cl::NDRange(n * channel * sample_size),
                              cl::NDRange(128)),
                    *(static_cast<const cl::Buffer *>(input->buffer())),
                    *(static_cast<cl::Buffer *>(scale->buffer())),
                    *(static_cast<cl::Buffer *>(offset->buffer())),
                    *(static_cast<cl::Buffer *>(mean->buffer())),
                    *(static_cast<cl::Buffer *>(var->buffer())),
                    *(static_cast<cl::Buffer *>(epsilon->buffer())),
                    static_cast<int>(channel),
                    static_cast<int>(sample_size),
                    *(static_cast<cl::Buffer *>(output->buffer())),
                    error);
  res_event.wait();
  MACE_CHECK(error == CL_SUCCESS);
}

}  // namespace kernels
}  //  namespace mace