//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/conv_2d.h"
#include "mace/core/common.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

void Conv1x1Naive(const Tensor *input,
                  const Tensor *filter,
                  const Tensor *bias,
                  Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t channels = output->dim(1);
  const index_t height = output->dim(2);
  const index_t width = output->dim(3);
  const index_t input_channels = input->dim(1);

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  auto conv_2d =
      cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int,
                        int>(program, "conv_2d_1x1_naive");
  const index_t pixels = height * width;

  cl_int error;
  conv_2d(cl::EnqueueArgs(
              runtime->command_queue(),
              cl::NDRange(static_cast<int>(batch), static_cast<int>(channels),
                          static_cast<int>(pixels)),
              cl::NDRange(1, 1, 128)),
          *(static_cast<cl::Buffer *>(input->buffer())),
          *(static_cast<cl::Buffer *>(filter->buffer())),
          *(static_cast<cl::Buffer *>(bias->buffer())),
          *(static_cast<cl::Buffer *>(output->buffer())),
          static_cast<int>(input_channels), error);
  MACE_CHECK(error == CL_SUCCESS);
}

void Conv1x1V2(const Tensor *input,
               const Tensor *filter,
               const Tensor *bias,
               Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t channels = output->dim(1);
  const index_t height = output->dim(2);
  const index_t width = output->dim(3);
  const index_t input_channels = input->dim(1);

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  const index_t pixels = height * width;
  const index_t channel_blocks = (channels + 3) / 4;
  const index_t pixel_blocks = (pixels + 3) / 4;

  // TODO KernelFunctor has an extra clReleaseCommandQueue due to a copy
  // TODO check wired clReleaseCommandQueue latency
  // The KernelFunctor can cause segment faults in cb_retain_event
  auto conv_2d_kernel = cl::Kernel(program, "conv_2d_1x1_v2");
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(conv_2d_kernel);
  uint32_t idx = 0;
  conv_2d_kernel.setArg(idx++,
                        *(static_cast<const cl::Buffer *>(input->buffer())));
  conv_2d_kernel.setArg(idx++,
                        *(static_cast<const cl::Buffer *>(filter->buffer())));
  if (bias == NULL) {
    conv_2d_kernel.setArg(idx++, NULL);
  } else {
    conv_2d_kernel.setArg(idx++,
                          *(static_cast<const cl::Buffer *>(bias->buffer())));
  }
  conv_2d_kernel.setArg(idx++, *(static_cast<cl::Buffer *>(output->buffer())));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input_channels));
  conv_2d_kernel.setArg(idx++, static_cast<int>(channels));
  conv_2d_kernel.setArg(idx++, static_cast<int>(pixels));

  auto command_queue = runtime->command_queue();
  cl_int error = command_queue.enqueueNDRangeKernel(
      conv_2d_kernel, cl::NullRange,
      cl::NDRange(static_cast<int>(batch), static_cast<int>(channel_blocks),
                  static_cast<int>(pixel_blocks)),
      cl::NDRange(1, 2, kwg_size / 2));
  MACE_CHECK(error == CL_SUCCESS, error);
}

void Conv1x1V3(const Tensor *input,
               const Tensor *filter,
               const Tensor *bias,
               Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t channels = output->dim(1);
  const index_t height = output->dim(2);
  const index_t width = output->dim(3);
  const index_t input_channels = input->dim(1);

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();

  const index_t pixels = height * width;
  const index_t pixel_blocks = (pixels + 3) / 4;

  const index_t channel_blocks = (channels + 3) / 4;
  const index_t input_channel_blocks = (input_channels + 3) / 4;

  // FIXME temp hacking
  static std::map<std::ptrdiff_t, cl::Image3D> input_image_map;
  static std::map<std::ptrdiff_t, cl::Image3D> output_image_map;
  cl::Image3D input_image;
  cl::Image3D output_image;
  auto input_iter =
      input_image_map.find(reinterpret_cast<std::ptrdiff_t>(input->buffer()));
  if (input_iter != input_image_map.end()) {
    input_image = input_iter->second;
  } else {
    // The batch dimension is collapsed with channel
    cl_int error;
    cl::Image3D image =
        cl::Image3D(OpenCLRuntime::Get()->context(),
                    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                    cl::ImageFormat(CL_RGBA, CL_FLOAT), height, width,
                    batch * input_channel_blocks, 0, 0, nullptr, &error);
    MACE_CHECK(error == CL_SUCCESS);
    input_image = image;
    input_image_map.clear();
    input_image_map.emplace(reinterpret_cast<std::ptrdiff_t>(input->buffer()),
                            image);
  }
  auto output_iter =
      output_image_map.find(reinterpret_cast<std::ptrdiff_t>(output->buffer()));
  if (output_iter != output_image_map.end()) {
    output_image = output_iter->second;
  } else {
    cl_int error;
    cl::Image3D image =
        cl::Image3D(OpenCLRuntime::Get()->context(),
                    CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                    cl::ImageFormat(CL_RGBA, CL_FLOAT), height, width,
                    batch * channel_blocks, 0, 0, nullptr, &error);
    MACE_CHECK(error == CL_SUCCESS);
    output_image = image;
    output_image_map.clear();
    output_image_map.emplace(reinterpret_cast<std::ptrdiff_t>(output->buffer()),
                             image);
  }

  auto conv_2d_kernel = cl::Kernel(program, "conv_2d_1x1_v3");
  const uint32_t kwg_size = runtime->GetKernelMaxWorkGroupSize(conv_2d_kernel);

  uint32_t idx = 0;
  conv_2d_kernel.setArg(idx++, input_image);
  conv_2d_kernel.setArg(idx++,
                        *(static_cast<const cl::Buffer *>(filter->buffer())));
  conv_2d_kernel.setArg(idx++,
                        *(static_cast<const cl::Buffer *>(bias->buffer())));
  conv_2d_kernel.setArg(idx++, output_image);
  conv_2d_kernel.setArg(idx++, static_cast<int>(batch));
  conv_2d_kernel.setArg(idx++, static_cast<int>(input_channels));
  conv_2d_kernel.setArg(idx++, static_cast<int>(channels));
  conv_2d_kernel.setArg(idx++, static_cast<int>(height));
  conv_2d_kernel.setArg(idx++, static_cast<int>(width));

  auto command_queue = runtime->command_queue();
  cl_int error;
  error = command_queue.enqueueNDRangeKernel(
      conv_2d_kernel, cl::NullRange,
      cl::NDRange(static_cast<int>(channel_blocks), static_cast<int>(height),
                  static_cast<int>(width)),
      cl::NDRange(1, 2, kwg_size / 2));
  MACE_CHECK(error == CL_SUCCESS, error);
}

extern void Conv2dOpenclK1x1S1(const Tensor *input,
                               const Tensor *filter,
                               const Tensor *bias,
                               Tensor *output) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(2);
  const index_t width = output->dim(3);

  const index_t input_batch = input->dim(0);
  const index_t input_height = input->dim(2);
  const index_t input_width = input->dim(3);

  MACE_CHECK(input_batch == batch && input_height == height &&
             input_width == width);

  Conv1x1V2(input, filter, bias, output);
};

}  // namespace kernels
}  // namespace mace
