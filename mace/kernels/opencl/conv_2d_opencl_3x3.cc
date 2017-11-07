//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/common.h"
#include "mace/core/macros.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/conv_2d.h"
#include "mace/kernels/opencl/space_to_batch.h"

namespace mace {
namespace kernels {


static void InnerConv2dK3x3S12(const Tensor *input, const Tensor *filter,
                               const Tensor *bias, const uint32_t stride,
                               Tensor *output, const std::vector<cl::Event> *waiting_events,
                               cl::Event *ret_event) {
  const index_t channels = output->shape()[1];
  const index_t height = output->shape()[2];
  const index_t width = output->shape()[3];

  MACE_CHECK(input->dim(0) == output->dim(0));

  const index_t channel_blocks = (channels + 3) / 4;
  const index_t pixel_blocks = (width + 3) / 4 * height;

  auto runtime = OpenCLRuntime::Get();
  auto program = runtime->program();
  auto conv_kernel = cl::Kernel(program, "conv_2d_3x3");

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
                           static_cast<uint32_t>(8),
                           static_cast<uint32_t>(128)};
  cl_int error = runtime->command_queue().enqueueNDRangeKernel(
      conv_kernel, cl::NullRange,
      cl::NDRange(gws[0], gws[1], gws[2]),
      cl::NDRange(lws[0], lws[1], lws[2]),
      waiting_events,
      ret_event);
  MACE_CHECK(error == CL_SUCCESS);
}

static void CalOutputShape(const std::vector<index_t> &input_shape,
                           const std::vector<index_t> &filter_shape,
                           const int dilation_height,
                           const int dilation_width,
                           std::vector<index_t> &output_shape) {
  index_t kernel_height = filter_shape[2];
  index_t kernel_width = filter_shape[3];
  index_t output_channels = filter_shape[0];

  index_t k_extent_height = (kernel_height - 1) * dilation_height + 1;
  index_t k_extent_width = (kernel_width - 1) * dilation_width + 1;
  index_t output_height = input_shape[2] - k_extent_height + 1;
  index_t output_width = input_shape[3] - k_extent_width + 1;
  output_shape[0] = input_shape[0];
  output_shape[1] = output_channels;
  output_shape[2] = output_height;
  output_shape[3] = output_width;
}
static void ResizeBatchTensor(const std::vector<index_t> &input_shape,
                              const int dilation_height,
                              const int dilation_width,
                              Tensor *batch_tensor) {
  LOG(INFO) << input_shape[2] << "\t" << input_shape[3] << "\t" <<dilation_height;
  batch_tensor->Resize({input_shape[0] * dilation_height * dilation_width,
                        input_shape[1],
                        input_shape[2] / dilation_height,
                        input_shape[3] / dilation_width}
  );
  LOG(INFO) << batch_tensor->dim(2) << "\t" << batch_tensor->dim(3) << "\t" <<dilation_width;
}

void Conv2dOpenclK3x3S1(const Tensor *input, const Tensor *filter,
                        const Tensor *bias, const int dilation_height,
                        const int dilation_width, Tensor *output) {
  if (dilation_height > 1 && dilation_width > 1) {
    cl::Event events[2];

    Tensor reshaped_input_tensor(GetDeviceAllocator(DeviceType::OPENCL), input->dtype());
    ResizeBatchTensor(input->shape(), dilation_height, dilation_width, &reshaped_input_tensor);
    SpaceToBatch(const_cast<Tensor*>(input), dilation_height, dilation_width,
                 &reshaped_input_tensor, nullptr, &events[0]);
    Tensor reshaped_output_tensor(GetDeviceAllocator(DeviceType::OPENCL), input->dtype());
    std::vector<index_t> reshaped_output_shape(4, 0);
    CalOutputShape(reshaped_input_tensor.shape(), filter->shape(),
                   dilation_height, dilation_width, reshaped_output_shape);
    reshaped_output_tensor.Resize(reshaped_output_shape);
    std::vector<cl::Event> s2b_events(1, events[0]);
    InnerConv2dK3x3S12(&reshaped_input_tensor, filter, bias, 1, &reshaped_output_tensor,
                       &s2b_events, &events[1]);
    std::vector<cl::Event> conv_events(1, events[1]);
    SpaceToBatch<true>(&reshaped_output_tensor, dilation_height, dilation_width,
                       output, &conv_events, nullptr);
  } else {
    InnerConv2dK3x3S12(input, filter, bias, 1, output, nullptr, nullptr);
  }
};

void Conv2dOpenclK3x3S2(const Tensor *input, const Tensor *filter,
                        const Tensor *bias, const int dilation_height,
                        const int dilation_width, Tensor *output) {
  MACE_UNUSED(dilation_height);
  MACE_UNUSED(dilation_width);
  InnerConv2dK3x3S12(input, filter, bias, 2, output, nullptr, nullptr);
};

}  // namespace kernels
}  // namespace mace
