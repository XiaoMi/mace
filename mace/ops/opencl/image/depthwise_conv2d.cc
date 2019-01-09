// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/opencl/image/depthwise_conv2d.h"

#include <algorithm>
#include <set>
#include <string>

namespace mace {
namespace ops {
namespace opencl {
namespace image {
namespace depthwise {

namespace {
// (inputs + weights + outputs) * array_size * sizeof(float)
const uint32_t kernel_cache_size = (4 + 4 + 1) * 4 * 4;
std::vector<uint32_t> LocalWS(OpenCLRuntime *runtime,
                              const uint32_t *gws,
                              const uint32_t kwg_size) {
  std::vector<uint32_t> lws(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint64_t
        cache_size = runtime->device_global_mem_cache_size();
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

MaceStatus DepthwiseConv2d(OpContext *context,
                           cl::Kernel *kernel,
                           const Tensor *input,   // NHWC
                           const Tensor *filter,  // HWIM
                           const Tensor *bias,
                           const int stride,
                           const int *paddings,
                           const int *dilations,
                           const ActivationType activation,
                           const float relux_max_limit,
                           const float leakyrelu_coefficient,
                           const DataType dt,
                           std::vector<index_t> *prev_input_shape,
                           Tensor *output,
                           uint32_t *kwg_size) {
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

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
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
      case LEAKYRELU:
        built_options.emplace("-DUSE_LEAKYRELU");
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
  MACE_OUT_OF_RANGE_INIT(*kernel);
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
    MACE_OUT_OF_RANGE_SET_ARGS(*kernel);
    MACE_SET_3D_GWS_ARGS(*kernel, gws);
    kernel->setArg(idx++, *(input->opencl_image()));
    kernel->setArg(idx++, *(filter->opencl_image()));
    if (bias != nullptr) {
      kernel->setArg(idx++, *(bias->opencl_image()));
    }
    kernel->setArg(idx++, *(output->opencl_image()));
    kernel->setArg(idx++, relux_max_limit);
    kernel->setArg(idx++, leakyrelu_coefficient);
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

  const std::vector<uint32_t> lws = LocalWS(runtime, gws, *kwg_size);
  std::string tuning_key =
      Concat("depthwise_conv2d_ocl_kernel", gws[0], gws[1], gws[2], multiplier);
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, *kernel, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace depthwise
}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
