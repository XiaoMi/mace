//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/common.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/conv_2d.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

static constexpr index_t kInputChannelBlockSize = 2;
static constexpr index_t kOutputChannelBlockSize = 4;

// TODO(heliangliang) fix bad performance
void AssignBias(Tensor *output, const Tensor *bias) {

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  if (bias == nullptr) {
    auto assign_bias =
      cl::KernelFunctor<cl::Buffer, float>(program, "assign_f32");
    int global_size = output->NumElements();
    cl_int error;
    assign_bias(cl::EnqueueArgs(runtime->command_queue(),
                                cl::NDRange(global_size),
                                cl::NullRange),
                *(static_cast<cl::Buffer *>(output->buffer())),
                0.0f, error);
    MACE_CHECK(error == CL_SUCCESS);
  } else {
    auto output_shape = output->shape();
    index_t batch = output_shape[0];
    index_t channels = output_shape[1];
    index_t pixels = output_shape[2] * output_shape[3];
    MACE_CHECK(channels == bias->shape()[0], "Channels mismatch");

    auto assign_bias =
      cl::KernelFunctor<cl::Buffer, cl::Buffer, int>(program, "assign_vec_f32");
    cl_int error;
    assign_bias(cl::EnqueueArgs(runtime->command_queue(),
                                cl::NDRange(batch, channels),
                                cl::NullRange),
                *(static_cast<cl::Buffer *>(output->buffer())),
                *(static_cast<cl::Buffer *>(bias->buffer())),
                static_cast<int>(pixels),
                error);
    MACE_CHECK(error == CL_SUCCESS);
  }
} 

extern void Conv2dOpenclK1x1S1(const Tensor *input, const Tensor *filter,
                               const Tensor *bias, Tensor *output) {
  const index_t batch = output->shape()[0];
  const index_t channels = output->shape()[1];
  const index_t height = output->shape()[2];
  const index_t width = output->shape()[3];

  const index_t input_batch = input->shape()[0];
  const index_t input_channels = input->shape()[1];
  const index_t input_height = input->shape()[2];
  const index_t input_width = input->shape()[3];

  MACE_CHECK(input_batch == batch && input_height == height &&
             input_width == width);

  AssignBias(output, bias);

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  auto conv_2d = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer,
                                   int, int, int, int, int>(program, "conv_2d_1x1_naive");
  const index_t total_pixels = height * width;

  for (int b = 0; b < batch; ++b) {
    int input_offset = b * input_channels * total_pixels;
    int output_offset = b * channels * total_pixels;
    int chan_blk_num = (channels + 3) >> 2; // each 4 output channels
    int pixel_blk_num = (total_pixels + 7) >> 3; // each 8 pixels
    cl_int error;
    conv_2d(cl::EnqueueArgs(runtime->command_queue(),
                            cl::NDRange(chan_blk_num, pixel_blk_num),
                            cl::NullRange),
            *(static_cast<cl::Buffer *>(input->buffer())),
            *(static_cast<cl::Buffer *>(filter->buffer())),
            *(static_cast<cl::Buffer *>(output->buffer())),
            input_offset, output_offset, total_pixels, input_channels, channels, error);
    MACE_CHECK(error == CL_SUCCESS);
  }
};

}  // namespace kernels
}  // namespace mace
