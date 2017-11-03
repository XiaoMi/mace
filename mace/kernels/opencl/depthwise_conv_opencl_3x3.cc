//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/conv_2d.h"

namespace mace {
namespace kernels {

static void InnerDepthwiseConvOpenclK3x3S12(const Tensor *input,
                                            const Tensor *filter,
                                            const Tensor *bias,
                                            const uint32_t stride,
                                            Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t channels = output->dim(1);
  const index_t height = output->dim(2);
  const index_t width = output->dim(3);

  const index_t input_batch = input->dim(0);
  const index_t input_channels = input->dim(1);
  const index_t input_height = input->dim(2);
  const index_t input_width = input->dim(3);

  MACE_CHECK(input_batch == batch);
  const index_t pixels = height * width;
  const index_t channel_blocks = (channels + 3) / 4;
  const index_t pixel_blocks = (width + 3) / 4 * height;

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  auto conv_kernel = cl::Kernel(program, "depthwise_conv_3x3");

  uint32_t idx = 0;
  conv_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(input->buffer())));
  conv_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(filter->buffer())));
  conv_kernel.setArg(idx++, *(static_cast<const cl::Buffer *>(bias->buffer())));
  conv_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));
  conv_kernel.setArg(idx++, static_cast<int32_t>(input->dim(1)));
  conv_kernel.setArg(idx++, static_cast<int32_t>(channels));
  conv_kernel.setArg(idx++, static_cast<int32_t>(input->dim(2)));
  conv_kernel.setArg(idx++, static_cast<int32_t>(input->dim(3)));
  conv_kernel.setArg(idx++, static_cast<int32_t>(height));
  conv_kernel.setArg(idx++, static_cast<int32_t>(width));
  conv_kernel.setArg(idx++, stride);
  conv_kernel.setArg(idx++, stride);

  const uint32_t gws[3] = {static_cast<uint32_t>(output->dim(0)),
                           static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(pixel_blocks)};
  const uint32_t lws[3] = {static_cast<uint32_t>(1),
                           static_cast<uint32_t>(1),
                           static_cast<uint32_t>(256)};
  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      conv_kernel, cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]));
  MACE_CHECK(error == CL_SUCCESS);
}

extern void DepthwiseConvOpenclK3x3S1(const Tensor *input,
                                      const Tensor *filter,
                                      const Tensor *bias,
                                      Tensor *output) {
  InnerDepthwiseConvOpenclK3x3S12(input, filter, bias, 1, output);
};

extern void DepthwiseConvOpenclK3x3S2(const Tensor *input,
                                      const Tensor *filter,
                                      const Tensor *bias,
                                      Tensor *output) {
  InnerDepthwiseConvOpenclK3x3S12(input, filter, bias, 2, output);
};

}  // namespace kernels
}  // namespace mace
