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

  const uint32_t gws[3] = {static_cast<uint32_t>(input->dim(0)),
                           static_cast<uint32_t>(input->dim(1)),
                           static_cast<uint32_t>(input->dim(2) * input->dim(3))};


  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  auto bm_kernel = cl::Kernel(program, "batch_norm");

  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(bm_kernel);
  const uint32_t lws[3] = {1, 1, kwg_size};

  uint32_t idx = 0;
  bm_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(scale->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(offset->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(mean->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(var->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(epsilon->buffer())));
  bm_kernel.setArg(idx++, gws[2]);
  bm_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));
  bm_kernel.setArg(idx++, lws[1] * sizeof(float), nullptr);
  bm_kernel.setArg(idx++, lws[1] * sizeof(float), nullptr);

  //TODO need to design the new way to tune.
  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      bm_kernel, cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]));
  MACE_CHECK(error == CL_SUCCESS);
}

}  // namespace kernels
}  //  namespace mace