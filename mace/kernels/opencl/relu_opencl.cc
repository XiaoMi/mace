//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/relu.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
void ReluFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                        Tensor *output) {

  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();

  std::set<std::string> built_options;
  auto dt = DataTypeToEnum<T>::value;
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  if (max_limit_ < 0) {
    auto relu_kernel  = runtime->BuildKernel("relu", "relu", built_options);
    const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(relu_kernel);
    const uint32_t lws[3] = {1, kwg_size, 1};

    uint32_t idx = 0;
    relu_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input->buffer())));
    relu_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));

    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        relu_kernel, cl::NullRange,
        cl::NDRange(gws[0], gws[1], gws[2]),
        cl::NDRange(lws[0], lws[1], lws[2]),
        NULL, OpenCLRuntime::Get()->GetDefaultEvent());
    MACE_CHECK(error == CL_SUCCESS);
  } else {
    auto relu_kernel  = runtime->BuildKernel("relu", "relux", built_options);
    const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(relu_kernel);
    const uint32_t lws[3] = {1, kwg_size, 1};

    uint32_t idx = 0;
    relu_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input->buffer())));
    relu_kernel.setArg(idx++, max_limit_);
    relu_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));

    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        relu_kernel, cl::NullRange,
        cl::NDRange(gws[0], gws[1], gws[2]),
        cl::NDRange(lws[0], lws[1], lws[2]),
        NULL, OpenCLRuntime::Get()->GetDefaultEvent());
    MACE_CHECK(error == CL_SUCCESS);
  }
}

template
struct ReluFunctor<DeviceType::OPENCL, float>;
template
struct ReluFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace
