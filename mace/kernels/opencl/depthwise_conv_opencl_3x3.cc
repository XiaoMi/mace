//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/conv_2d.h"

namespace mace {
namespace kernels {

extern void DepthwiseConvOpenclK3x3S1(const Tensor *input,
                                      const Tensor *filter,
                                      const Tensor *bias,
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

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  auto conv_2d = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                                   int, int, int, int, int, int, int>(program, "depthwise_conv_3x3_s1");
  const index_t pixels = height * width;
  const index_t channel_blocks = (channels + 3) / 4;
  const index_t pixel_blocks = (width + 3) / 4 * height;

  cl_int error;
  conv_2d(cl::EnqueueArgs(runtime->command_queue(),
                          cl::NDRange(static_cast<int>(batch),
                                      static_cast<int>(channel_blocks),
                                      static_cast<int>(pixel_blocks)),
                          cl::NDRange(1, 1, 256)),
          *(static_cast<cl::Buffer *>(input->buffer())),
          *(static_cast<cl::Buffer *>(filter->buffer())),
          *(static_cast<cl::Buffer *>(bias->buffer())),
          *(static_cast<cl::Buffer *>(output->buffer())),
          static_cast<int>(input_channels),
          static_cast<int>(channels),
          static_cast<int>(input_height),
          static_cast<int>(input_width),
          static_cast<int>(height),
          static_cast<int>(width),
          error);
  MACE_CHECK(error == CL_SUCCESS);
};

}  // namespace kernels
}  // namespace mace
