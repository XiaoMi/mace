//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/depthwise_conv2d.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

void DepthwiseConv2d(const Tensor *input,   // NHWC
                     const Tensor *filter,  // HWIM
                     const Tensor *bias,
                     const int stride,
                     const int *paddings,
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

  const index_t input_batch = input->dim(0);
  const index_t input_height = input->dim(1);
  const index_t input_width = input->dim(2);
  const index_t input_channels = input->dim(3);

  const index_t filter_height = filter->dim(0);
  const index_t filter_width = filter->dim(1);
  const index_t multiplier = filter->dim(3);
  MACE_CHECK(multiplier == 1, "Multiplier > 1 not supported");
  MACE_CHECK(multiplier * input_channels == channels);
  MACE_CHECK(filter->dim(2) == input_channels, filter->dim(2), "!=",
             input_channels);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const index_t width_blocks = RoundUpDiv4(width);

  auto runtime = OpenCLRuntime::Global();
  std::set<std::string> built_options;
  std::string kernel_name = MACE_OBFUSCATE_SYMBOL("depthwise_conv2d");
  built_options.emplace("-Ddepthwise_conv2d=" + kernel_name);
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

  auto dw_conv2d_kernel =
      runtime->BuildKernel("depthwise_conv2d", kernel_name, built_options);

  uint32_t idx = 0;
  dw_conv2d_kernel.setArg(idx++,
                          *(static_cast<const cl::Image2D *>(input->buffer())));
  dw_conv2d_kernel.setArg(
      idx++, *(static_cast<const cl::Image2D *>(filter->buffer())));
  if (bias != nullptr) {
    dw_conv2d_kernel.setArg(
        idx++, *(static_cast<const cl::Image2D *>(bias->buffer())));
  }
  dw_conv2d_kernel.setArg(
      idx++, *(static_cast<const cl::Image2D *>(output->buffer())));
  dw_conv2d_kernel.setArg(idx++, relux_max_limit);
  dw_conv2d_kernel.setArg(idx++, prelu_alpha);
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(input_height));
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(input_width));
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(input_channel_blocks));
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(height));
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(width));
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(filter_height));
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(filter_width));
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(paddings[0] / 2));
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(paddings[1] / 2));
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(dilations[0]));
  dw_conv2d_kernel.setArg(idx++, static_cast<short>(dilations[1]));

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};
  const std::vector<uint32_t> lws = {8, 16, 8, 1};
  std::string tuning_key = Concat("depthwise_conv2d_ocl_kernel_", activation,
                                  batch, height, width, channels, multiplier);
  TuningOrRun3DKernel(dw_conv2d_kernel, tuning_key, gws, lws, future);
}

template <typename T>
void DepthwiseConv2dFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    Tensor *output,
    StatsFuture *future) {
  typedef void (*Conv2dOpenclFunction)(const Tensor *input,
                                       const Tensor *filter, const Tensor *bias,
                                       Tensor *output, StatsFuture *future);
  index_t kernel_h = filter->dim(2);
  index_t kernel_w = filter->dim(3);
  if (strides_[0] != strides_[1]) {
    LOG(WARNING) << "OpenCL depthwise conv2d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides_[0] << "x" << strides_[1]
                 << " is not implemented yet, using slow version";
    // TODO(heliangliang) The CPU/NEON kernel should map the buffer
    DepthwiseConv2dFunctor<DeviceType::CPU, float>(
        strides_, padding_, dilations_, activation_, relux_max_limit_,
        prelu_alpha_)(input, filter, bias, output, future);
    return;
  }

  // Create a fake conv_2d filter to calculate the paddings and output size
  std::vector<index_t> fake_filter_shape(4);
  fake_filter_shape[0] = filter->shape()[0];
  fake_filter_shape[1] = filter->shape()[1];
  fake_filter_shape[3] = filter->shape()[2] * filter->shape()[3];
  fake_filter_shape[2] = 1;

  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  kernels::CalcNHWCPaddingAndOutputSize(
      input->shape().data(), fake_filter_shape.data(), dilations_, strides_,
      padding_, output_shape.data(), paddings.data());

  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, output_image_shape);
  output->ResizeImage(output_shape, output_image_shape);

  DepthwiseConv2d(input, filter, bias, strides_[0], paddings.data(), dilations_,
                  activation_, relux_max_limit_, prelu_alpha_,
                  DataTypeToEnum<T>::value, output, future);
}

template struct DepthwiseConv2dFunctor<DeviceType::OPENCL, float>;
template struct DepthwiseConv2dFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace
