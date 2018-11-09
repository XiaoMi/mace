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
#ifndef MACE_OPS_OPENCL_IMAGE_DEPTHWISE_DECONV2D_H_
#define MACE_OPS_OPENCL_IMAGE_DEPTHWISE_DECONV2D_H_

#include "mace/ops/opencl/depthwise_deconv2d.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class DepthwiseDeconv2dKernel : public OpenCLDepthwiseDeconv2dKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const Tensor *filter,
      const Tensor *bias,
      const int *strides,
      const int *padding_data,
      const int group,
      const ActivationType activation,
      const float relux_max_limit,
      const std::vector<index_t> &output_shape,
      Tensor *output) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus DepthwiseDeconv2dKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *filter,
    const Tensor *bias,
    const int *strides,
    const int *padding_data,
    const int group,
    const ActivationType activation,
    const float relux_max_limit,
    const std::vector<index_t> &output_shape,
    Tensor *output) {
  const index_t batch = output_shape[0];
  const index_t height = output_shape[1];
  const index_t width = output_shape[2];
  const index_t channels = output_shape[3];
  const index_t input_channels = input->dim(3);
  const index_t multiplier = filter->dim(0);

  MACE_CHECK(group == channels && group == input_channels && multiplier == 1,
             "opencl image deconv only supports depthwise type group.");

  std::vector<size_t> output_image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
                  &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));
  const DataType dt = DataTypeToEnum<T>::value;

  const index_t channel_blocks = RoundUpDiv4(channels);
  const int stride_h = strides[0];
  const int stride_w = strides[1];
  MACE_CHECK(stride_w > 0 && stride_h > 0, "strides should be > 0.");
  const int width_tile = 5;
  const index_t n_strides = (width + stride_w - 1) / stride_w;
  const index_t width_blocks =
      ((n_strides + width_tile - 1) / width_tile) * stride_w;
  const float stride_h_r = 1.f / static_cast<float>(stride_h);
  const float stride_w_r = 1.f / static_cast<float>(stride_w);
  const int padding_h = (padding_data[0] + 1) >> 1;
  const int padding_w = (padding_data[1] + 1) >> 1;

  const int align_h = stride_h - 1 - padding_h;
  const int align_w = stride_w - 1 - padding_w;
  const int kernel_size = filter->dim(2) * filter->dim(3);

  auto runtime = context->device()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("depthwise_deconv2d");
    built_options.emplace("-Ddepthwise_deconv2d=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
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

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("depthwise_deconv2d", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width_blocks),
                           static_cast<uint32_t>(height * batch)};

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(filter->opencl_image()));
    if (bias != nullptr) {
      kernel_.setArg(idx++, *(bias->opencl_image()));
    }
    kernel_.setArg(idx++, *(output->opencl_image()));
    kernel_.setArg(idx++, relux_max_limit);
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(input->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(height));
    kernel_.setArg(idx++, static_cast<int32_t>(width));
    kernel_.setArg(idx++, static_cast<int32_t>(channels));
    kernel_.setArg(idx++, static_cast<int32_t>(stride_h));
    kernel_.setArg(idx++, static_cast<int32_t>(stride_w));
    kernel_.setArg(idx++, stride_h_r);
    kernel_.setArg(idx++, stride_w_r);
    kernel_.setArg(idx++, static_cast<int32_t>(align_h));
    kernel_.setArg(idx++, static_cast<int32_t>(align_w));
    kernel_.setArg(idx++, static_cast<int32_t>(padding_h));
    kernel_.setArg(idx++, static_cast<int32_t>(padding_w));
    kernel_.setArg(idx++, static_cast<int32_t>(filter->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(filter->dim(3)));
    kernel_.setArg(idx++, static_cast<int32_t>(kernel_size));
    kernel_.setArg(idx++, static_cast<int32_t>(channel_blocks));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("depthwise_deconv2d_kernel_",
             activation,
             output->dim(0),
             output->dim(1),
             output->dim(2),
             output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_DEPTHWISE_DECONV2D_H_
