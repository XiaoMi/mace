//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/relu.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void ReluFunctor<DeviceType::OPENCL, T>::operator()(const Tensor *input,
                                                    Tensor *output,
                                                    StatsFuture *future) {

  const index_t batch = input->dim(0);
  const index_t height = input->dim(1);
  const index_t width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  auto runtime = OpenCLRuntime::Global();

  std::set<std::string> built_options;
  auto dt = DataTypeToEnum<T>::value;
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  cl::Kernel relu_kernel;
  if (max_limit_ < 0) {
    std::string kernel_name = MACE_KERNRL_NAME("relu");
    built_options.emplace("-Drelu=" + kernel_name);
    relu_kernel  = runtime->BuildKernel("relu", kernel_name, built_options);

    uint32_t idx = 0;
    relu_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
    relu_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));
  } else {
    std::string kernel_name = MACE_KERNRL_NAME("relux");
    built_options.emplace("-Drelux=" + kernel_name);
    relu_kernel  = runtime->BuildKernel("relu", kernel_name, built_options);

    uint32_t idx = 0;
    relu_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
    relu_kernel.setArg(idx++, max_limit_);
    relu_kernel.setArg(idx++, *(static_cast<cl::Image2D *>(output->buffer())));
  }
  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  const std::vector<uint32_t> lws = {8, 16, 8};
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(relu_kernel);
  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(3, 0);
    local_ws[0] = std::min<uint32_t>(channel_blocks, kwg_size);
    local_ws[1] = std::min<uint32_t>(width, kwg_size / local_ws[0]);
    local_ws[2] = std::min<uint32_t>(height * batch, kwg_size / (local_ws[0] * local_ws[1]));
    return {{local_ws[0], local_ws[1], local_ws[2]},
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
            {4, 15, 8}, //SNPE size
    };
  };
  cl::Event event;
  auto func = [&](const std::vector<uint32_t> &params) -> cl_int {
    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        relu_kernel, cl::NullRange,
        cl::NDRange(gws[0], gws[1], gws[2]),
        cl::NDRange(params[0], params[1], params[2]),
        nullptr, &event);

    MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
    return error;
  };
  std::stringstream ss;
  ss << "relu_opencl_kernel_"
     << output->dim(0) << "_"
     << output->dim(1) << "_"
     << output->dim(2) << "_"
     << output->dim(3);
  OpenCLProfilingTimer timer(&event);
  Tuner<uint32_t>::Get()->template TuneOrRun<cl_int>(ss.str(),
                                                     lws,
                                                     params_generator,
                                                     func,
                                                     &timer);
  if (future != nullptr) {
    future->wait_fn = [runtime, event](CallStats *stats) {
      event.wait();
      if (stats != nullptr) {
        runtime->GetCallStats(event, stats);
      }
    };
  }
}

template
struct ReluFunctor<DeviceType::OPENCL, float>;
template
struct ReluFunctor<DeviceType::OPENCL, half>;
}  // namespace kernels
}  // namespace mace
