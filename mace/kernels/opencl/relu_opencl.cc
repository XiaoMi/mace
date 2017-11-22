//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/relu.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"

namespace mace {
namespace kernels {

template <>
void ReluFunctor<DeviceType::OPENCL, float>::operator()(const Tensor *input,
                                                        Tensor *output) {

  index_t element_size = input->NumElements();
  index_t blocks = (element_size + 3) / 4;

  const uint32_t gws = blocks;

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();

  std::set<std::string> built_options;
  built_options.emplace("-DDATA_TYPE=" + DataTypeToCLType(input->dtype()));
  if (max_limit_ < 0) {
    auto relu_kernel  = runtime->BuildKernel("relu", "relu", built_options);
    const uint32_t lws = runtime->GetKernelMaxWorkGroupSize(relu_kernel);

    uint32_t idx = 0;
    relu_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input->buffer())));
    relu_kernel.setArg(idx++, static_cast<int32_t>(element_size));
    relu_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));

    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        relu_kernel, cl::NullRange,
        cl::NDRange(gws),
        cl::NDRange(lws),
        NULL, OpenCLRuntime::GetDefaultEvent());
    MACE_CHECK(error == CL_SUCCESS);
  } else {
    auto relu_kernel  = runtime->BuildKernel("relu", "relux", built_options);

    const uint32_t lws = runtime->GetKernelMaxWorkGroupSize(relu_kernel);

    uint32_t idx = 0;
    relu_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input->buffer())));
    relu_kernel.setArg(idx++, max_limit_);
    relu_kernel.setArg(idx++, static_cast<int32_t>(element_size));
    relu_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));

    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        relu_kernel, cl::NullRange,
        cl::NDRange(gws),
        cl::NDRange(lws),
        NULL, OpenCLRuntime::GetDefaultEvent());
    MACE_CHECK(error == CL_SUCCESS);
  }
}

}  // namespace kernels
}  // namespace mace
