//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/activation.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
void ActivationFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                          Tensor *output,
                                                          StatsFuture *future) {
  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  auto runtime = OpenCLRuntime::Global();

  std::set<std::string> built_options;
  std::string kernel_name = MACE_OBFUSCATE_SYMBOL("activation");
  built_options.emplace("-Dactivation=" + kernel_name);
  auto dt = DataTypeToEnum<T>::value;
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  switch (activation_) {
    case RELU:
      built_options.emplace("-DUSE_RELU");
      break;
    case RELUX:
      built_options.emplace("-DUSE_RELUX");
      break;
    case PRELU:
      built_options.emplace("-DUSE_PRELU");
      break;
    case TANH:
      built_options.emplace("-DUSE_TANH");
      break;
    case SIGMOID:
      built_options.emplace("-DUSE_SIGMOID");
      break;
    defeult:
      LOG(FATAL) << "Unknown activation type: " << activation_;
  }
  cl::Kernel activation_kernel =
      runtime->BuildKernel("activation", kernel_name, built_options);
  int idx = 0;
  activation_kernel.setArg(
      idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
  activation_kernel.setArg(idx++, relux_max_limit_);
  activation_kernel.setArg(idx++, prelu_alpha_);
  activation_kernel.setArg(idx++,
                           *(static_cast<cl::Image2D *>(output->buffer())));

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  const std::vector<uint32_t> lws = {8, 16, 8};
  const uint32_t kwg_size =
      runtime->GetKernelMaxWorkGroupSize(activation_kernel);
  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(3, 0);
    local_ws[0] = std::min<uint32_t>(channel_blocks, kwg_size);
    local_ws[1] = std::min<uint32_t>(width, kwg_size / local_ws[0]);
    local_ws[2] = std::min<uint32_t>(height * batch,
                                     kwg_size / (local_ws[0] * local_ws[1]));
    return {
        {local_ws[0], local_ws[1], local_ws[2]},
        {kwg_size / 16, 4, 4},
        {kwg_size / 32, 4, 8},
        {kwg_size / 32, 8, 4},
        {kwg_size / 64, 8, 8},
        {kwg_size / 64, 16, 4},
        {kwg_size / 128, 8, 16},
        {kwg_size / 128, 16, 8},
        {kwg_size / 128, 32, 4},
        {1, kwg_size / 32, 32},
        {1, kwg_size / 64, 64},
        {1, kwg_size / 128, 128},
        {3, 15, 9},
        {7, 15, 9},
        {9, 7, 15},
        {15, 7, 9},
        {1, kwg_size, 1},
        {4, 15, 8},  // SNPE size
    };
  };
  cl::Event event;
  auto func = [&](const std::vector<uint32_t> &params) -> cl_int {
    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        activation_kernel, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
        cl::NDRange(params[0], params[1], params[2]), nullptr, &event);

    MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
    return error;
  };
  std::string tuning_key =
      Concat("relu_opencl_kernel_", activation_, output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  OpenCLProfilingTimer timer(&event);
  Tuner<uint32_t>::Get()->template TuneOrRun<cl_int>(
      tuning_key, lws, params_generator, func, &timer);
  SetFuture(future, event);
}

template struct ActivationFunctor<DeviceType::OPENCL, float>;
template struct ActivationFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace
