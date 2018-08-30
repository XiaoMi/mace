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

#include "mace/kernels/winograd_transform.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
MaceStatus WinogradTransformFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input_tensor, Tensor *output_tensor, StatsFuture *future) {
  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name;
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_);
    NON_UNIFORM_WG_CONFIG;
    if (wino_blk_size_ == 4) {
      obfuscated_kernel_name =
          MACE_OBFUSCATE_SYMBOL("winograd_transform_4x4");
      built_options.emplace("-Dwinograd_transform_4x4="
                                + obfuscated_kernel_name);
    } else if (wino_blk_size_ == 2) {
      obfuscated_kernel_name =
          MACE_OBFUSCATE_SYMBOL("winograd_transform_2x2");
      built_options.emplace("-Dwinograd_transform_2x2="
                                + obfuscated_kernel_name);
    } else {
      MACE_CHECK(false, "mace only supports 4x4 and 2x2 gpu winograd.");
      return MACE_SUCCESS;
    }
    built_options.emplace("-DDATA_TYPE=" +
                          DtToUpCompatibleCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToUpCompatibleCLCMDDt(DataTypeToEnum<T>::value));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("winograd_transform",
                                              obfuscated_kernel_name,
                                              built_options,
                                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  std::vector<index_t> output_shape(4);
  std::vector<index_t> filter_shape = {1, input_tensor->dim(3), 3, 3};
  std::vector<int> paddings(2);
  if (paddings_.empty()) {
    kernels::CalcNHWCPaddingAndOutputSize(
        input_tensor->shape().data(), filter_shape.data(), dilations_.data(),
        strides_.data(), padding_type_, output_shape.data(), paddings.data());
  } else {
    paddings = paddings_;
    CalcOutputSize(input_tensor->shape().data(), filter_shape.data(),
                   paddings_.data(), dilations_.data(), strides_.data(),
                   RoundType::FLOOR, output_shape.data());
  }
  const index_t round_h =
      (output_shape[1] + wino_blk_size_ - 1) / wino_blk_size_;
  const index_t round_w =
      (output_shape[2] + wino_blk_size_ - 1) / wino_blk_size_;
  const index_t out_width = input_tensor->dim(0) * round_h * round_w;

  const float round_hw_r = 1.f / static_cast<float>(round_h * round_w);
  const float round_w_r = 1.f / static_cast<float>(round_w);
  const index_t blk_sqr = (wino_blk_size_ + 2) * (wino_blk_size_ + 2);

  const uint32_t gws[2] = {
      static_cast<uint32_t>(out_width),
      static_cast<uint32_t>(RoundUpDiv4(input_tensor->dim(3)))
  };
  if (!IsVecEqual(input_shape_, input_tensor->shape())) {
    output_shape = {blk_sqr, input_tensor->dim(3), out_width};
    std::vector<index_t> padded_output_shape = {
        output_shape[0], output_shape[1], output_shape[2], 1
    };
    std::vector<size_t> image_shape;
    CalImage2DShape(padded_output_shape,
                    BufferType::IN_OUT_HEIGHT,
                    &image_shape);
    // remove unused last dimension
    MACE_RETURN_IF_ERROR(output_tensor->ResizeImage(output_shape, image_shape));

    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_2D_GWS_ARGS(kernel_);
    kernel_.setArg(idx++, *(input_tensor->opencl_image()));
    kernel_.setArg(idx++, *(output_tensor->opencl_image()));
    kernel_.setArg(idx++, static_cast<uint32_t>(input_tensor->dim(1)));
    kernel_.setArg(idx++, static_cast<uint32_t>(input_tensor->dim(2)));
    kernel_.setArg(idx++, static_cast<uint32_t>(input_tensor->dim(3)));
    kernel_.setArg(idx++, static_cast<uint32_t>(round_h * round_w));
    kernel_.setArg(idx++, round_hw_r);
    kernel_.setArg(idx++, static_cast<uint32_t>(round_w));
    kernel_.setArg(idx++, round_w_r);
    kernel_.setArg(idx++, static_cast<uint32_t>(paddings[0] / 2));
    kernel_.setArg(idx++, static_cast<uint32_t>(paddings[1] / 2));

    input_shape_ = input_tensor->shape();
  }


  const std::vector<uint32_t> lws = {kwg_size_ / 8, 8, 0};
  std::string tuning_key = Concat("winograd_transform_kernel",
                                  output_tensor->dim(0),
                                  output_tensor->dim(1),
                                  output_tensor->dim(2));
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(kernel_, tuning_key,
                                           gws, lws, future));

  OUT_OF_RANGE_VALIDATION(kernel_error_);
  return MACE_SUCCESS;
}

template <typename T>
MaceStatus WinogradInverseTransformFunctor<DeviceType::GPU, T>::operator()(
    const std::vector<const Tensor*> &inputs,
    Tensor *output_tensor,
    StatsFuture *future) {
  auto runtime = OpenCLRuntime::Global();

  const Tensor *input_tensor = inputs[0];
  const Tensor *bias = inputs.size() == 3 ? inputs[2] : nullptr;

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name;
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_);
    NON_UNIFORM_WG_CONFIG;
    if (wino_blk_size_ == 4) {
      obfuscated_kernel_name =
          MACE_OBFUSCATE_SYMBOL("winograd_inverse_transform_4x4");
      built_options.emplace("-Dwinograd_inverse_transform_4x4="
                                + obfuscated_kernel_name);
    } else if (wino_blk_size_ == 2) {
      obfuscated_kernel_name =
          MACE_OBFUSCATE_SYMBOL("winograd_inverse_transform_2x2");
      built_options.emplace("-Dwinograd_inverse_transform_2x2="
                                + obfuscated_kernel_name);
    } else {
      MACE_CHECK(false, "mace only supports 4x4 and 2x2 gpu winograd.");
      return MACE_SUCCESS;
    }

    built_options.emplace("-DDATA_TYPE=" +
                          DtToUpCompatibleCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToUpCompatibleCLCMDDt(DataTypeToEnum<T>::value));
    built_options.emplace(bias != nullptr ? "-DBIAS" : "");
    switch (activation_) {
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
      default:
        LOG(FATAL) << "Unknown activation type: " << activation_;
    }

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("winograd_transform",
                                              obfuscated_kernel_name,
                                              built_options,
                                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  Tensor::MappingGuard output_shape_guard(inputs[1]);
  const int32_t *output_shape_data = inputs[1]->data<int32_t>();
  const index_t batch = output_shape_data[0];
  const index_t height = output_shape_data[1];
  const index_t width = output_shape_data[2];
  const uint32_t gws[2] = {
      static_cast<uint32_t>(input_tensor->dim(2)),
      static_cast<uint32_t>(RoundUpDiv4(input_tensor->dim(1)))};
  if (!IsVecEqual(input_shape_, input_tensor->shape())) {
    std::vector<index_t> output_shape = {batch, height, width,
                                         input_tensor->dim(1)};
    std::vector<size_t> image_shape;
    CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);
    MACE_RETURN_IF_ERROR(output_tensor->ResizeImage(output_shape, image_shape));

    const index_t round_h = (height + wino_blk_size_ - 1) / wino_blk_size_;
    const index_t round_w = (width + wino_blk_size_ - 1) / wino_blk_size_;

    const float round_hw_r = 1.f / static_cast<float>(round_h * round_w);
    const float round_w_r = 1.f / static_cast<float>(round_w);

    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_2D_GWS_ARGS(kernel_);
    kernel_.setArg(
        idx++,
        *(static_cast<const cl::Image2D *>(input_tensor->opencl_image())));
    if (bias != nullptr) {
      kernel_.setArg(idx++,
                     *(static_cast<const cl::Image2D *>(bias->opencl_image())));
    }
    kernel_.setArg(
        idx++, *(static_cast<cl::Image2D *>(output_tensor->opencl_image())));
    kernel_.setArg(idx++, static_cast<uint32_t>(output_shape[1]));
    kernel_.setArg(idx++, static_cast<uint32_t>(output_shape[2]));
    kernel_.setArg(idx++, static_cast<uint32_t>(round_h * round_w));
    kernel_.setArg(idx++, round_hw_r);
    kernel_.setArg(idx++, static_cast<uint32_t>(round_w));
    kernel_.setArg(idx++, round_w_r);
    kernel_.setArg(idx++, relux_max_limit_);

    input_shape_ = input_tensor->shape();
  }
  const std::vector<uint32_t> lws = {kwg_size_ / 8, 8, 0};
  std::string tuning_key =
      Concat("winograd_inverse_transform_kernel", output_tensor->dim(0),
             output_tensor->dim(1), output_tensor->dim(2),
             output_tensor->dim(3), input_tensor->dim(2));
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(kernel_, tuning_key,
                                           gws, lws, future));

  OUT_OF_RANGE_VALIDATION(kernel_error_);
  return MACE_SUCCESS;
}

template struct WinogradTransformFunctor<DeviceType::GPU, float>;
template struct WinogradTransformFunctor<DeviceType::GPU, half>;

template struct WinogradInverseTransformFunctor<DeviceType::GPU, float>;
template struct WinogradInverseTransformFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
