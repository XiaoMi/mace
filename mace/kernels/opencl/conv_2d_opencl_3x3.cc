//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/conv_2d.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

static void Conv2d3x3S12(const Tensor *input, const Tensor *filter,
                         const Tensor *bias, const bool fused_relu,
                         const uint32_t stride, const int *padding,
                         const int *dilations, const DataType dt,
                         Tensor *output, StatsFuture *future) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const index_t width_blocks = RoundUpDiv<index_t, 5>(width);

  std::set<std::string> built_options;
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  built_options.emplace(bias != nullptr ? "-DBIAS" : "");
  built_options.emplace("-DSTRIDE=" + ToString(stride));
  if (fused_relu) {
    built_options.emplace("-DFUSED_RELU");
  }

  auto runtime = OpenCLRuntime::Global();
  auto conv_2d_kernel = runtime->BuildKernel("conv_2d_3x3", "conv_2d_3x3", built_options);

  uint32_t idx = 0;
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(filter->buffer())));
  if (bias != nullptr) {
    conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(bias->buffer())));
  }
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(output->buffer())));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input->dim(1)));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input->dim(2)));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input_channel_blocks));
  conv_2d_kernel.setArg(idx++, static_cast<int>(height));
  conv_2d_kernel.setArg(idx++, static_cast<int>(width));
  conv_2d_kernel.setArg(idx++, padding[0] / 2);
  conv_2d_kernel.setArg(idx++, padding[1] / 2);
  conv_2d_kernel.setArg(idx++, dilations[0]);
  conv_2d_kernel.setArg(idx++, dilations[1]);

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};
  const std::vector<uint32_t> lws = {4, 15, 8};
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(conv_2d_kernel);
  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(3, 0);
    local_ws[0] = std::min<uint32_t>(channel_blocks, kwg_size);
    local_ws[1] = std::min<uint32_t>(width_blocks, kwg_size / local_ws[0]);
    local_ws[2] = std::min<uint32_t>(height * batch, kwg_size / (local_ws[0] * local_ws[1]));
    return {{4, 15, 8}, //SNPE size
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
            {1, kwg_size, 1}};
  };
  cl::Event event;
  auto func = [&](const std::vector<uint32_t> &params) -> cl_int {
    cl_int error = runtime->command_queue().enqueueNDRangeKernel(
        conv_2d_kernel, cl::NullRange,
        cl::NDRange(gws[0], gws[1], gws[2]),
        cl::NDRange(params[0], params[1], params[2]),
        nullptr, &event);

    MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
    return error;
  };
  std::stringstream ss;
  ss << "conv2d_3x3_opencl_kernel_"
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
void Conv2dOpenclK3x3S1(const Tensor *input,
                        const Tensor *filter,
                        const Tensor *bias,
                        const bool fused_relu,
                        const int *padding,
                        const int *dilations,
                        const DataType dt,
                        Tensor *output,
                        StatsFuture *future) {
  Conv2d3x3S12(input, filter, bias, fused_relu, 1, padding, dilations, dt, output, future);
};

void Conv2dOpenclK3x3S2(const Tensor *input,
                        const Tensor *filter,
                        const Tensor *bias,
                        const bool fused_relu,
                        const int *padding,
                        const int *dilations,
                        const DataType dt,
                        Tensor *output,
                        StatsFuture *future) {
  Conv2d3x3S12(input, filter, bias, fused_relu, 2, padding, dilations, dt, output, future);
};

}  // namespace kernels
}  // namespace mace
