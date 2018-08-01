// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/activation.h"
#include "mace/kernels/depthwise_conv2d.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

namespace {
// (inputs + weights + outputs) * array_size * sizeof(float)
const uint32_t kernel_cache_size = (4 + 4 + 1) * 4 * 4;
std::vector<uint32_t> LocalWS(const uint32_t *gws, const uint32_t kwg_size) {
  std::vector<uint32_t> lws(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint64_t
        cache_size = OpenCLRuntime::Global()->device_global_mem_cache_size();
    uint32_t base = cache_size / kBaseGPUMemCacheSize;
    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    if (lws[1] >= base) {
      lws[0] = std::min<uint32_t>(gws[0], base);
    } else {
      lws[0] = std::min<uint32_t>(gws[0] / 8, kwg_size / lws[1]);
      if (lws[0] < base) {
        lws[0] = std::min<uint32_t>(std::max<uint32_t>(gws[0] / 4, base),
                                    kwg_size / lws[1]);
      }
    }
    lws[0] =
        std::max<uint32_t>(std::min<uint32_t>(lws[0], kwg_size / lws[1]), 1);
    const uint32_t lws_size = lws[0] * lws[1];
    lws[2] = std::min<uint32_t>((cache_size / kernel_cache_size / lws_size) * 4,
                                gws[2]);
    if (lws[2] == 0) {
      lws[2] = gws[2];
    }
    lws[2] = std::max<uint32_t>(std::min<uint32_t>(lws[2], kwg_size / lws_size),
                                1);
  }
  return lws;
}

}  // namespace

static MaceStatus DepthwiseConv2d(cl::Kernel *kernel,
                                  const Tensor *input,   // NHWC
                                  const Tensor *filter,  // HWIM
                                  const Tensor *bias,
                                  const int stride,
                                  const int *paddings,
                                  const int *dilations,
                                  const ActivationType activation,
                                  const float relux_max_limit,
                                  const DataType dt,
                                  std::vector<index_t> *prev_input_shape,
                                  Tensor *output,
                                  StatsFuture *future,
                                  uint32_t *kwg_size,
                                  std::unique_ptr<BufferBase> *kernel_error) {
  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);

  const index_t input_channels = input->dim(3);
  const index_t multiplier = filter->dim(0);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t input_channel_blocks = RoundUpDiv4(input_channels);
  const index_t width_blocks = RoundUpDiv4(width);

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};

  auto runtime = OpenCLRuntime::Global();

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(*kernel_error);
    NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("depthwise_conv2d");
    if (stride == 1 && dilations[0] == 1 && dilations[1] == 1) {
      kernel_name = MACE_OBFUSCATE_SYMBOL("depthwise_conv2d_s1");
      built_options.emplace("-Ddepthwise_conv2d_s1=" + kernel_name);
    } else {
      built_options.emplace("-Ddepthwise_conv2d=" + kernel_name);
    }
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
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

    MACE_RETURN_IF_ERROR(
        runtime->BuildKernel("depthwise_conv2d", kernel_name,
                             built_options, kernel));

    *kwg_size =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
  }
  if (!IsVecEqual(*prev_input_shape, input->shape())) {
    const index_t input_height = input->dim(1);
    const index_t input_width = input->dim(2);

    const index_t filter_height = filter->dim(2);
    const index_t filter_width = filter->dim(3);
    MACE_CHECK(multiplier == 1, "Multiplier > 1 not supported");
    MACE_CHECK(multiplier * input_channels == channels);
    MACE_CHECK(filter->dim(1) == input_channels, filter->dim(1), "!=",
               input_channels);

    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG_PTR;
    SET_3D_GWS_ARGS_PTR(kernel, gws);
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(filter->opencl_image()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_image()));
    }
    kernel->setArg(idx++, *(output->opencl_image()));
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, static_cast<int16_t>(input_height));
    kernel->setArg(idx++, static_cast<int16_t>(input_width));
    kernel->setArg(idx++, static_cast<int16_t>(input_channel_blocks));
    kernel->setArg(idx++, static_cast<int16_t>(height));
    kernel->setArg(idx++, static_cast<int16_t>(width));
    kernel->setArg(idx++, static_cast<int16_t>(filter_height));
    kernel->setArg(idx++, static_cast<int16_t>(filter_width));
    kernel->setArg(idx++, static_cast<int16_t>(paddings[0] / 2));
    kernel->setArg(idx++, static_cast<int16_t>(paddings[1] / 2));
    if (stride != 1 || dilations[0] != 1 || dilations[1] != 1) {
      kernel->setArg(idx++, static_cast<int16_t>(dilations[0]));
      kernel->setArg(idx++, static_cast<int16_t>(dilations[1]));
    }

    *prev_input_shape = input->shape();
  }

  const std::vector<uint32_t> lws = LocalWS(gws, *kwg_size);
  std::string tuning_key =
      Concat("depthwise_conv2d_ocl_kernel", gws[0], gws[1], gws[2], multiplier);
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(*kernel, tuning_key,
                                           gws, lws, future));

  OUT_OF_RANGE_VALIDATION(*kernel_error);
  return MACE_SUCCESS;
}

template <typename T>
MaceStatus DepthwiseConv2dFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const Tensor *filter, /* MIHW */
    const Tensor *bias,
    Tensor *output,
    StatsFuture *future) {
  index_t kernel_h = filter->dim(2);
  index_t kernel_w = filter->dim(3);
  if (strides_[0] != strides_[1]) {
    LOG(WARNING) << "OpenCL depthwise conv2d kernel with "
                 << "filter" << kernel_h << "x" << kernel_w << ","
                 << " stride " << strides_[0] << "x" << strides_[1]
                 << " is not implemented yet, using slow version";
    // TODO(heliangliang) The CPU/NEON kernel should map the buffer
    return DepthwiseConv2dFunctor<DeviceType::CPU, float>(
        strides_, padding_type_, paddings_, dilations_, activation_,
        relux_max_limit_)(input, filter, bias, output, future);
  }

  // Create a fake conv_2d filter to calculate the paddings and output size
  std::vector<index_t> fake_filter_shape(4);
  fake_filter_shape[0] = filter->dim(0) * filter->dim(1);
  fake_filter_shape[1] = filter->dim(1);
  fake_filter_shape[2] = filter->dim(2);
  fake_filter_shape[3] = filter->dim(3);

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
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                  &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  return DepthwiseConv2d(
      &kernel_, input, filter, bias, strides_[0], paddings.data(), dilations_,
      activation_, relux_max_limit_, DataTypeToEnum<T>::value, &input_shape_,
      output, future, &kwg_size_, &kernel_error_);
}

template struct DepthwiseConv2dFunctor<DeviceType::GPU, float>;
template struct DepthwiseConv2dFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
