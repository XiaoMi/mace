//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/utils/utils.h"
#include "mace/kernels/opencl/helper.h"

namespace mace {
namespace kernels {

void Conv1x1V2(const Tensor *input,
               const Tensor *filter,
               const Tensor *bias,
               const int stride,
               Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t channels = output->dim(1);
  const index_t height = output->dim(2);
  const index_t width = output->dim(3);
  const index_t input_channels = input->dim(1);

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  const index_t channel_blocks = (channels + 3) / 4;
  const index_t pixel_blocks = (width + 3) / 4 * height;

  // TODO KernelFunctor has an extra clReleaseCommandQueue due to a copy
  // TODO check wired clReleaseCommandQueue latency
  // The KernelFunctor can cause segment faults in cb_retain_event
  std::set<std::string> built_options;
  built_options.emplace("-DDATA_TYPE=" + DataTypeToCLType(input->dtype()));
  built_options.emplace(stride == 1 ? "-DSTRIDE_1" : "");
  built_options.emplace(bias != nullptr ? "-DBIAS" : "");
  auto conv_2d_kernel = runtime->BuildKernel("conv_2d_1x1", "conv_2d_1x1_v2", built_options);

  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(conv_2d_kernel);
  uint32_t idx = 0;
  conv_2d_kernel.setArg(idx++,
                        *(static_cast<const cl::Buffer *>(input->buffer())));
  conv_2d_kernel.setArg(idx++,
                        *(static_cast<const cl::Buffer *>(filter->buffer())));
  if (bias != nullptr) {
    conv_2d_kernel.setArg(idx++,
                          *(static_cast<const cl::Buffer *>(bias->buffer())));
  }
  conv_2d_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input_channels));
  conv_2d_kernel.setArg(idx++, static_cast<int>(channels));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input->dim(2)));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input->dim(3)));
  conv_2d_kernel.setArg(idx++, static_cast<int>(height));
  conv_2d_kernel.setArg(idx++, static_cast<int>(width));

  auto command_queue = runtime->command_queue();
  cl_int error = command_queue.enqueueNDRangeKernel(
      conv_2d_kernel, cl::NullRange,
      cl::NDRange(static_cast<int>(batch), static_cast<int>(channel_blocks),
                  static_cast<int>(pixel_blocks)),
      cl::NDRange(1, 2, kwg_size / 2),
      NULL, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS, error);
}

void Conv1x1V3(const Tensor *input,
               const Tensor *filter,
               const Tensor *bias,
               const int stride,
               Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t channels = output->dim(1);
  const index_t height = output->dim(2);
  const index_t width = output->dim(3);
  const index_t input_channels = input->dim(1);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);

  std::set<std::string> built_options;
  built_options.emplace("-DDATA_TYPE=" + DataTypeToCLType(input->dtype()));
  built_options.emplace("-DSTRIDE_1");
  built_options.emplace(bias != nullptr ? "-DBIAS" : "");

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();

  auto conv_2d_kernel = runtime->BuildKernel("conv_2d_1x1", "conv_2d_1x1", built_options);
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(conv_2d_kernel);

  uint32_t idx = 0;
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(input->buffer())));
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(filter->buffer())));
  if (bias != nullptr) {
    conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(bias->buffer())));
  }
  conv_2d_kernel.setArg(idx++, *(static_cast<const cl::Image2D *>(output->buffer())));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input_channel_blocks));
  conv_2d_kernel.setArg(idx++, static_cast<int>(width));

  auto command_queue = runtime->command_queue();
  cl_int error;
  error = command_queue.enqueueNDRangeKernel(
      conv_2d_kernel, cl::NullRange,
      cl::NDRange(static_cast<uint32_t>(channel_blocks), static_cast<uint32_t>(height),
                  static_cast<uint32_t>(height * batch)),
      cl::NDRange(4, 15, 8),
      NULL, OpenCLRuntime::Get()->GetDefaultEvent());
  MACE_CHECK(error == CL_SUCCESS, error);
}

extern void Conv2dOpenclK1x1S1(const Tensor *input,
                               const Tensor *filter,
                               const Tensor *bias,
                               const int *padding,
                               Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(2);
  const index_t width = output->dim(3);

  const index_t input_batch = input->dim(0);
  const index_t input_height = input->dim(2);
  const index_t input_width = input->dim(3);

  MACE_CHECK(input_batch == batch && input_height == height &&
             input_width == width);

  Conv1x1V2(input, filter, bias, 1, output);
};

extern void Conv2dOpenclK1x1S2(const Tensor *input,
                               const Tensor *filter,
                               const Tensor *bias,
                               const int *padding,
                               Tensor *output) {
  MACE_CHECK(input->dim(0) == output->dim(0));

  Conv1x1V2(input, filter, bias, 2, output);
};

}  // namespace kernels
}  // namespace mace
