//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/conv_2d.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

void Conv1x1Naive(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output) {
  const index_t batch = output->shape()[0];
  const index_t channels = output->shape()[1];
  const index_t height = output->shape()[2];
  const index_t width = output->shape()[3];
  const index_t input_channels = input->shape()[1];

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  auto conv_2d = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                                   int, int>(program, "conv_2d_1x1_naive");
  const index_t pixels = height * width;

  cl_int error;
  conv_2d(cl::EnqueueArgs(runtime->command_queue(),
                          cl::NDRange(static_cast<int>(batch),
                                      static_cast<int>(channels),
                                      static_cast<int>(pixels)),
                          cl::NDRange(1, 1, 128)),
          *(static_cast<cl::Buffer *>(input->buffer())),
          *(static_cast<cl::Buffer *>(filter->buffer())),
          *(static_cast<cl::Buffer *>(bias->buffer())),
          *(static_cast<cl::Buffer *>(output->buffer())),
          static_cast<int>(input_channels),
          error);
  MACE_CHECK(error == CL_SUCCESS);
}

void Conv1x1V2(const Tensor *input,
               const Tensor *filter,
               const Tensor *bias,
               Tensor *output) {
  const index_t batch = output->shape()[0];
  const index_t channels = output->shape()[1];
  const index_t height = output->shape()[2];
  const index_t width = output->shape()[3];
  const index_t input_channels = input->shape()[1];

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  auto conv_2d = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                                   int, int, int, int>(program, "conv_2d_1x1_v2");
  const index_t pixels = height * width;
  const index_t channel_blocks = (channels + 3) / 4;
  const index_t pixel_blocks = (pixels + 3) / 4;

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
          static_cast<int>(pixels),
          error);
  MACE_CHECK(error == CL_SUCCESS);
}

extern void Conv2dOpenclK1x1S1(const Tensor *input, const Tensor *filter,
                               const Tensor *bias, Tensor *output) {
  const index_t batch = output->shape()[0];
  const index_t height = output->shape()[2];
  const index_t width = output->shape()[3];

  const index_t input_batch = input->shape()[0];
  const index_t input_height = input->shape()[2];
  const index_t input_width = input->shape()[3];

  MACE_CHECK(input_batch == batch && input_height == height &&
             input_width == width);

  Conv1x1V2(input, filter, bias, output);
};

}  // namespace kernels
}  // namespace mace
