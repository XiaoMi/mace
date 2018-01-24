//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

void Conv2dOpencl(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  const uint32_t stride,
                  const int *padding,
                  const int *dilations,
                  const ActivationType activation,
                  const float relux_max_limit,
                  const float prelu_alpha,
                  const DataType dt,
                  Tensor *output,
                  StatsFuture *future) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);
  const index_t input_channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const index_t width_blocks = RoundUpDiv4(width);

  std::set<std::string> built_options;
  std::string kernel_name = MACE_OBFUSCATE_SYMBOL("conv_2d");
  built_options.emplace("-Dconv_2d=" + kernel_name);
  built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
  built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
  built_options.emplace(bias != nullptr ? "-DBIAS" : "");
  built_options.emplace("-DSTRIDE=" + ToString(stride));
  switch (activation) {
    case NOOP:
      break;
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
      LOG(FATAL) << "Unknown activation type: " << activation;
  }

  auto runtime = OpenCLRuntime::Global();
  auto conv_2d_kernel =
      runtime->BuildKernel("conv_2d", kernel_name, built_options);

  uint32_t idx = 0;
  conv_2d_kernel.setArg(idx++,
                        *(static_cast<const cl::Image2D *>(input->buffer())));
  conv_2d_kernel.setArg(idx++,
                        *(static_cast<const cl::Image2D *>(filter->buffer())));
  if (bias != nullptr) {
    conv_2d_kernel.setArg(idx++,
                          *(static_cast<const cl::Image2D *>(bias->buffer())));
  }
  conv_2d_kernel.setArg(idx++,
                        *(static_cast<const cl::Image2D *>(output->buffer())));
  conv_2d_kernel.setArg(idx++, relux_max_limit);
  conv_2d_kernel.setArg(idx++, prelu_alpha);
  conv_2d_kernel.setArg(idx++, static_cast<int>(input->dim(1)));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input->dim(2)));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input_channel_blocks));
  conv_2d_kernel.setArg(idx++, static_cast<int>(height));
  conv_2d_kernel.setArg(idx++, static_cast<int>(width));
  conv_2d_kernel.setArg(idx++, static_cast<int>(filter->dim(0)));
  conv_2d_kernel.setArg(idx++, static_cast<int>(filter->dim(1)));
  conv_2d_kernel.setArg(idx++, padding[0] / 2);
  conv_2d_kernel.setArg(idx++, padding[1] / 2);
  conv_2d_kernel.setArg(idx++, dilations[0]);
  conv_2d_kernel.setArg(idx++, dilations[1]);

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};
  std::vector<uint32_t> lws = {8, 16, 8, 1};
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(conv_2d_kernel);
  auto params_generator = [&]() -> std::vector<std::vector<uint32_t>> {
    std::vector<uint32_t> local_ws(3, 0);
    local_ws[0] = std::min<uint32_t>(channel_blocks, kwg_size);
    local_ws[1] = std::min<uint32_t>(width_blocks, kwg_size / local_ws[0]);
    local_ws[2] = std::min<uint32_t>(height * batch,
                                     kwg_size / (local_ws[0] * local_ws[1]));
    return {
        {local_ws[0], local_ws[1], local_ws[2], 1},
        {kwg_size / 16, 4, 4, 1},
        {kwg_size / 32, 4, 8, 1},
        {kwg_size / 32, 8, 4, 1},
        {kwg_size / 64, 8, 8, 1},
        {kwg_size / 64, 16, 4, 1},
        {kwg_size / 128, 8, 16, 1},
        {kwg_size / 128, 16, 8, 1},
        {kwg_size / 128, 32, 4, 1},
        {1, kwg_size / 32, 32, 1},
        {1, kwg_size / 64, 64, 1},
        {1, kwg_size / 128, 128, 1},
        {3, 15, 9, 1},
        {7, 15, 9, 1},
        {9, 7, 15, 1},
        {15, 7, 9, 1},
        {1, kwg_size, 1, 1},
        {4, 15, 8, 1},  // SNPE size
    };
  };
  cl::Event event;
  auto func = [&](std::vector<uint32_t> &params, Timer *timer) -> cl_int {
    cl_int error = CL_SUCCESS;
    if (timer == nullptr) {
      uint32_t num_blocks = params.back();
      const uint32_t block_size = gws[2] / num_blocks;
      if (gws[2] % num_blocks > 0) num_blocks++;
      for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t gws2 = (i == num_blocks - 1) ? (gws[2] - (i * block_size)) : block_size;
        error = runtime->command_queue().enqueueNDRangeKernel(
            conv_2d_kernel,
            cl::NDRange(0, 0, i * block_size),
            cl::NDRange(gws[0], gws[1], gws2),
            cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
        MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
      }
    } else {
      timer->StartTiming();
      error = runtime->command_queue().enqueueNDRangeKernel(
          conv_2d_kernel, cl::NullRange, cl::NDRange(gws[0], gws[1], gws[2]),
          cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
      MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
      timer->StopTiming();
      double elapse_time = timer->ElapsedMicros();
      timer->ClearTiming();
      uint32_t num_blocks = std::min(static_cast<uint32_t>(elapse_time / kMaxKernelExeTime) + 1, gws[2]);
      params.back() = num_blocks;
      const uint32_t block_size = gws[2] / num_blocks;
      if (gws[2] % num_blocks > 0) num_blocks++;
      for (uint32_t i = 0; i < num_blocks; ++i) {
        uint32_t gws2 = (i == num_blocks - 1) ? (gws[2] - (i * block_size)) : block_size;
        error = runtime->command_queue().enqueueNDRangeKernel(
            conv_2d_kernel,
            cl::NDRange(0, 0, i * block_size),
            cl::NDRange(gws[0], gws[1], gws2),
            cl::NDRange(params[0], params[1], params[2]), nullptr, &event);
        MACE_CHECK(error == CL_SUCCESS) << "Error code: " << error;
        timer->AccumulateTiming();
      }
    }
    return error;
  };
  std::string tuning_key =
      Concat("conv2d_general_opencl_kernel_", activation, output->dim(0),
             output->dim(1), output->dim(2), output->dim(3));
  OpenCLProfilingTimer timer(&event);
  Tuner<uint32_t>::Get()->template TuneOrRun<cl_int>(
      tuning_key, lws, params_generator, func, &timer);
  SetFuture(future, event);
}

}  // namespace kernels
}  // namespace mace
