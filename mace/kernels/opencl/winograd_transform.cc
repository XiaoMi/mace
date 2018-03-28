//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/winograd_transform.h"
#include "mace/core/runtime/opencl/cl2_header.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
void WinogradTransformFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input_tensor, Tensor *output_tensor, StatsFuture *future) {

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    is_non_uniform_work_groups_supported_ =
        runtime->IsNonUniformWorkgroupsSupported();
    std::string obfuscated_kernel_name =
        MACE_OBFUSCATE_SYMBOL("winograd_transform_2x2");
    std::set<std::string> built_options;
    built_options.emplace("-Dwinograd_transform_2x2=" + obfuscated_kernel_name);
    built_options.emplace("-DDATA_TYPE=" +
                          DtToUpstreamCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToUpstreamCLCMDDt(DataTypeToEnum<T>::value));
    if (is_non_uniform_work_groups_supported_) {
      built_options.emplace("-DUSE_QUALCOMM_OPENCL_2_0");
    }
    kernel_ = runtime->BuildKernel("winograd_transform", obfuscated_kernel_name,
                                   built_options);
  }
  std::vector<index_t> output_shape(4);
  std::vector<index_t> filter_shape = {3, 3, input_tensor->dim(3), 1};
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
  const index_t round_h = (output_shape[1] + 1) / 2;
  const index_t round_w = (output_shape[2] + 1) / 2;
  const index_t out_width = input_tensor->dim(0) * round_h * round_w;
  const uint32_t gws[2] = {
      static_cast<uint32_t>(out_width),
      static_cast<uint32_t>(RoundUpDiv4(input_tensor->dim(3)))};

  if (!IsVecEqual(input_shape_, input_tensor->shape())) {
    output_shape = {16, input_tensor->dim(3), out_width, 1};
    std::vector<size_t> image_shape;
    CalImage2DShape(output_shape, BufferType::IN_OUT_HEIGHT, &image_shape);
    output_tensor->ResizeImage(output_shape, image_shape);

    uint32_t idx = 0;
    kernel_.setArg(idx++, *(input_tensor->opencl_image()));
    kernel_.setArg(idx++, *(output_tensor->opencl_image()));
    kernel_.setArg(idx++, static_cast<uint32_t>(input_tensor->dim(1)));
    kernel_.setArg(idx++, static_cast<uint32_t>(input_tensor->dim(2)));
    kernel_.setArg(idx++, static_cast<uint32_t>(input_tensor->dim(3)));
    kernel_.setArg(idx++, static_cast<uint32_t>(round_h * round_w));
    kernel_.setArg(idx++, static_cast<uint32_t>(round_w));
    kernel_.setArg(idx++, static_cast<uint32_t>(paddings[0] / 2));
    kernel_.setArg(idx++, static_cast<uint32_t>(paddings[1] / 2));
    kernel_.setArg(idx++, gws[0]);
    kernel_.setArg(idx++, gws[1]);

    input_shape_ = input_tensor->shape();

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const std::vector<uint32_t> lws = {kwg_size_ / 8, 8, 1};
  std::stringstream ss;
  ss << "winograd_transform_kernel_" << input_tensor->dim(0) << "_"
     << input_tensor->dim(1) << "_" << input_tensor->dim(2) << "_"
     << input_tensor->dim(3);
  TuningOrRun2DKernel(kernel_, ss.str(), gws, lws, future);
}

template <typename T>
void WinogradInverseTransformFunctor<DeviceType::OPENCL, T>::operator()(
    const Tensor *input_tensor,
    const Tensor *bias,
    Tensor *output_tensor,
    StatsFuture *future) {

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    is_non_uniform_work_groups_supported_ =
        runtime->IsNonUniformWorkgroupsSupported();
    std::string obfuscated_kernel_name =
        MACE_OBFUSCATE_SYMBOL("winograd_inverse_transform_2x2");
    std::set<std::string> built_options;
    built_options.emplace("-Dwinograd_inverse_transform_2x2=" +
                          obfuscated_kernel_name);
    built_options.emplace("-DDATA_TYPE=" +
                          DtToUpstreamCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
                          DtToUpstreamCLCMDDt(DataTypeToEnum<T>::value));
    if (is_non_uniform_work_groups_supported_) {
      built_options.emplace("-DUSE_QUALCOMM_OPENCL_2_0");
    }
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

    kernel_ = runtime->BuildKernel("winograd_transform", obfuscated_kernel_name,
                                   built_options);
  }

  const uint32_t gws[2] = {
      static_cast<uint32_t>(input_tensor->dim(2)),
      static_cast<uint32_t>(RoundUpDiv4(input_tensor->dim(1)))};
  if (!IsVecEqual(input_shape_, input_tensor->shape())) {
    std::vector<index_t> output_shape = {batch_, height_, width_,
                                         input_tensor->dim(1)};
    std::vector<size_t> image_shape;
    CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);
    output_tensor->ResizeImage(output_shape, image_shape);

    const uint32_t round_h = (height_ + 1) / 2;
    const uint32_t round_w = (width_ + 1) / 2;
    uint32_t idx = 0;
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
    kernel_.setArg(idx++, static_cast<uint32_t>(round_w));
    kernel_.setArg(idx++, relux_max_limit_);
    kernel_.setArg(idx++, gws[0]);
    kernel_.setArg(idx++, gws[1]);

    input_shape_ = input_tensor->shape();

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const std::vector<uint32_t> lws = {kwg_size_ / 8, 8, 1};

  std::stringstream ss;
  ss << "winograd_inverse_transform_kernel_" << input_tensor->dim(0) << "_"
     << input_tensor->dim(1) << "_" << input_tensor->dim(2) << "_"
     << input_tensor->dim(3);
  TuningOrRun2DKernel(kernel_, ss.str(), gws, lws, future);
}

template struct WinogradTransformFunctor<DeviceType::OPENCL, float>;
template struct WinogradTransformFunctor<DeviceType::OPENCL, half>;

template struct WinogradInverseTransformFunctor<DeviceType::OPENCL, float>;
template struct WinogradInverseTransformFunctor<DeviceType::OPENCL, half>;

}  // namespace kernels
}  // namespace mace
