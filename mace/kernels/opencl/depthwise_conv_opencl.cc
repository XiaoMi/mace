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

void DepthwiseConv2d(cl::Kernel *kernel,
                     const Tensor *input,   // NHWC
                     const Tensor *filter,  // HWIM
                     const Tensor *bias,
                     const int stride,
                     const int *paddings,
                     const int *dilations,
                     const ActivationType activation,
                     const float relux_max_limit,
                     const DataType dt,
                     Tensor *output,
                     StatsFuture *future) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);

  const index_t input_channels = input->dim(3);
  const index_t multiplier = filter->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const index_t width_blocks = RoundUpDiv4(width);
  if (kernel->get() == nullptr) {
    const index_t input_batch = input->dim(0);
    const index_t input_height = input->dim(1);
    const index_t input_width = input->dim(2);

    const index_t filter_height = filter->dim(0);
    const index_t filter_width = filter->dim(1);
    MACE_CHECK(multiplier == 1, "Multiplier > 1 not supported");
    MACE_CHECK(multiplier * input_channels == channels);
    MACE_CHECK(filter->dim(2) == input_channels, filter->dim(2), "!=",
               input_channels);

    auto runtime = OpenCLRuntime::Global();
    std::set<std::string> built_options;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("depthwise_conv2d");
    if (stride == 1 && dilations[0] == 1 && dilations[1] == 1) {
      kernel_name = MACE_OBFUSCATE_SYMBOL("depthwise_conv2d_s1");
      built_options.emplace("-Ddepthwise_conv2d_s1=" + kernel_name);
    } else {
      built_options.emplace("-Ddepthwise_conv2d=" + kernel_name);
    }
    built_options.emplace("-DDATA_TYPE=" + DtToUpstreamCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpstreamCLCMDDt(dt));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    built_options.emplace(MakeString("-DSTRIDE=", stride));
    switch (activation) {
      case NOOP:
        break;
      case RELU:
        built_options.emplace("-DUSE_RELU");
        break;
      case RELUX:
        built_options.emplace("-DUSE_RELUX");
        break;
      case TANH:
        built_options.emplace("-DUSE_TANH");
        break;
      case SIGMOID:
        built_options.emplace("-DUSE_SIGMOID");
        break;
      default:
        LOG(FATAL) << "Unknown activation type: " << activation;
    }

    *kernel =
        runtime->BuildKernel("depthwise_conv2d", kernel_name, built_options);

    uint32_t idx = 0;
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(filter->opencl_image()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_image()));
    }
    kernel->setArg(idx++, *(output->opencl_image()));
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, static_cast<short>(input_height));
    kernel->setArg(idx++, static_cast<short>(input_width));
    kernel->setArg(idx++, static_cast<short>(input_channel_blocks));
    kernel->setArg(idx++, static_cast<short>(height));
    kernel->setArg(idx++, static_cast<short>(width));
    kernel->setArg(idx++, static_cast<short>(filter_height));
    kernel->setArg(idx++, static_cast<short>(filter_width));
    kernel->setArg(idx++, static_cast<short>(paddings[0] / 2));
    kernel->setArg(idx++, static_cast<short>(paddings[1] / 2));
    if (stride != 1 || dilations[0] != 1 || dilations[1] != 1) {
      kernel->setArg(idx++, static_cast<short>(dilations[0]));
      kernel->setArg(idx++, static_cast<short>(dilations[1]));
    }
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};
  const std::vector<uint32_t> lws = {8, 16, 8, 1};
  std::string tuning_key = Concat("depthwise_conv2d_ocl_kernel_", activation,
                                  batch, height, width, channels, multiplier);
  TuningOrRun3DKernel(*kernel, tuning_key, gws, lws, future);
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
        strides_, padding_type_, paddings_, dilations_, activation_,
        relux_max_limit_)(input, filter, bias, output, future);
    return;
  }

  // Create a fake conv_2d filter to calculate the paddings and output size
  std::vector<index_t> fake_filter_shape(4);
  fake_filter_shape[0] = filter->shape()[0];
  fake_filter_shape[1] = filter->shape()[1];
  fake_filter_shape[2] = filter->shape()[2] * filter->shape()[3];
  fake_filter_shape[3] = 1;

  std::vector<index_t> output_shape(4);
  std::vector<int> paddings(2);
  if (paddings_.empty()) {
    kernels::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), fake_filter_shape.data(), dilations_, strides_,
        padding_type_, output_shape.data(), paddings.data());
  } else {
    paddings = paddings_;
    CalcOutputSize(input->shape().data(), fake_filter_shape.data(),
                   paddings_.data(), dilations_, strides_, RoundType::FLOOR,
                   output_shape.data());
  }

  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, output_image_shape);
  output->ResizeImage(output_shape, output_image_shape);

  DepthwiseConv2d(&kernel_, input, filter, bias, strides_[0], paddings.data(),
                  dilations_, activation_, relux_max_limit_,
                  DataTypeToEnum<T>::value, output, future);
}

template struct DepthwiseConv2dFunctor<DeviceType::OPENCL, float>;
template struct DepthwiseConv2dFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace
