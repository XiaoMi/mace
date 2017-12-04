//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/batch_norm.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/utils/tuner.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

template <typename T>
void BatchNormFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input,
    const Tensor *scale,
    const Tensor *offset,
    const Tensor *mean,
    const Tensor *var,
    const Tensor *epsilon,
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
  std::set<std::string> built_options;
  auto dt = DataTypeToEnum<T>::value;
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  auto bm_kernel = runtime->BuildKernel("batch_norm", "batch_norm", built_options);

  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(bm_kernel);
  const std::vector<uint32_t> lws = {1, kwg_size, 1};

  uint32_t idx = 0;
  bm_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(scale->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(offset->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(mean->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(var->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(epsilon->buffer())));
  bm_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));

  auto params_generator = [&kwg_size]()->std::vector<std::vector<uint32_t>> {
    return {{8, 128, 1}, //SNPE size
            {1, 1, 64},
            {1, 1, 128},
            {1, kwg_size/16, 16},
            {1, kwg_size/32, 32},
            {1, kwg_size/64, 64},
            {1, kwg_size/128, 128},
            {1, 1, kwg_size},
            {1, kwg_size, 1}};
  };
  auto func = [&](const std::vector<uint32_t>& params)->cl_int {
    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        bm_kernel, cl::NullRange,
        cl::NDRange(gws[0], gws[1], gws[2]),
        cl::NDRange(params[0], params[1], params[2]),
        NULL, OpenCLRuntime::Get()->GetDefaultEvent());

    MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
    return error;
  };
  std::stringstream ss;
  ss << "batch_norm_opencl_kernel_"
      << input->dim(0) << "_"
      << input->dim(1) << "_"
      << input->dim(2) << "_"
      << input->dim(3);
  Tuner<uint32_t>::Get()->template TuneOrRun<cl_int>(ss.str(),
                                                     lws,
                                                     params_generator,
                                                     func);
}

template
struct BatchNormFunctor<DeviceType::OPENCL, float>;
template
struct BatchNormFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace
