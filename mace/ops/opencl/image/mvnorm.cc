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

#include "mace/ops/opencl/image/mvnorm.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace mace {
namespace ops {
namespace opencl {
namespace image {

namespace {

MaceStatus BuildMVNKernel(OpenCLRuntime *runtime, cl::Kernel *kernel,
                          const char *kernel_name,
                          std::set<std::string> *built_options,
                          bool across_channel) {
  std::stringstream micro_name;
  micro_name << "-Dmvnorm=" << kernel_name;
  built_options->emplace(micro_name.str());
  built_options->emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
  built_options->emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
  if (across_channel) {
    built_options->emplace("-DACROSS_CHANNELS");
  }
  MACE_RETURN_IF_ERROR(runtime->BuildKernel("mvnorm", kernel_name,
                                            *built_options, kernel));
  return MaceStatus::MACE_SUCCESS;
}

std::unique_ptr<Image> CreateImage(
    OpContext *context, const DataType dt,
    const std::vector<index_t> &buffer_shape) {
  std::unique_ptr<Image> image =
      make_unique<Image>(context->device()->allocator());
  std::vector<size_t> shape;
  OpenCLUtil::CalImage2DShape(
      buffer_shape, OpenCLBufferType::IN_OUT_CHANNEL, &shape);
  MACE_CHECK(image->Allocate(shape, dt) == MaceStatus::MACE_SUCCESS);
  VLOG(1) << "MVNormKernel::CreateImage allocate image_:" << MakeString(shape);

  return image;
}

}  // namespace

MVNormKernel::MVNormKernel(bool normalize_variance,
                           bool across_channels, float eps)
    : normalize_variance_(normalize_variance),
      across_channels_(across_channels),
      eps_(eps) {}

void MVNormKernel::CheckImage(OpContext *context, const DataType dt,
                              const std::vector<index_t> &square_shape,
                              const std::vector<index_t> &mean_shape) {
  if (square_image_ == nullptr) {
    square_image_ = CreateImage(context, dt, square_shape);
  }

  if (mean_image_ == nullptr) {
    mean_image_ = CreateImage(context, dt, mean_shape);
  }
}

MaceStatus MVNormKernel::Compute(OpContext
                                 *context,
                                 const Tensor *input, Tensor
                                 *output) {
  const auto batch = input->dim(0);
  const auto height = input->dim(1);
  const auto width = input->dim(2);
  const auto channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  auto runtime = context->device()->gpu_runtime()->opencl_runtime();

  if (normalize_variance_) {
    const std::vector<index_t> &square_shape = input->buffer_shape();
    const std::vector<index_t> mean_shape = {1, 1, 1, channels};
    CheckImage(context, input->dtype(), square_shape, mean_shape);
    // compute the (X - EX)^2
    MACE_RETURN_IF_ERROR(ExecuteVarianceNormStep1Kernel(
        context, runtime, gws, input));
    // compute the compute (X - EX) / (E((X - EX)^2)^0.5 + eps_)
    MACE_RETURN_IF_ERROR(ExecuteVarianceNormStep2Kernel(
        context, runtime, gws, input, output));
  } else {
    MACE_RETURN_IF_ERROR(ExecuteMeanNormKernel(
        context, runtime, gws, input, output));
  }

  return
      MaceStatus::MACE_SUCCESS;
}

MaceStatus MVNormKernel::ExecuteMeanNormKernel(OpContext *context,
                                               OpenCLRuntime *runtime,
                                               const uint32_t (&gws)[3],
                                               const Tensor *input,
                                               Tensor *output) {
  const auto height = input->dim(1);
  MACE_OUT_OF_RANGE_DEFINITION;
  if (kernel_step1_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    MACE_RETURN_IF_ERROR(BuildMVNKernel(runtime, &kernel_step1_, "mvnorm_mean",
                                        &built_options, across_channels_));
    kwg_size_step1_ = static_cast<uint32_t>(
        runtime->GetKernelMaxWorkGroupSize(kernel_step1_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_step1_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_step1_);
  MACE_SET_3D_GWS_ARGS(kernel_step1_, gws);
  kernel_step1_.setArg(idx++, *(input->opencl_image()));
  kernel_step1_.setArg(idx++, static_cast<int>(height));
  kernel_step1_.setArg(idx++, *(output->opencl_image()));

  std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_step1_);
  std::string
      tuning_key = Concat("mvnorm_mean_opencl_kernel", gws[0], gws[1], gws[2],
                          normalize_variance_, across_channels_);

  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_step1_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

// The first step of compute Variance Norm, compute the (X - EX)^2
// store them into the square_image_
MaceStatus MVNormKernel::ExecuteVarianceNormStep1Kernel(
    OpContext *context, OpenCLRuntime *runtime,
    const uint32_t (&gws)[3], const Tensor *input) {
  const auto height = input->dim(1);
  MACE_OUT_OF_RANGE_DEFINITION;
  if (kernel_step1_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    MACE_RETURN_IF_ERROR(BuildMVNKernel(runtime, &kernel_step1_,
                                        "mvnorm_vn_step1",
                                        &built_options, across_channels_));
    kwg_size_step1_ = static_cast<uint32_t>(
        runtime->GetKernelMaxWorkGroupSize(kernel_step1_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_step1_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_step1_);
  MACE_SET_3D_GWS_ARGS(kernel_step1_, gws);
  kernel_step1_.setArg(idx++, *(input->opencl_image()));
  cl::Image *mean_image = static_cast<cl::Image *>(mean_image_->buffer());
  kernel_step1_.setArg(idx++, *mean_image);
  cl::Image *square_image = static_cast<cl::Image *>(square_image_->buffer());
  kernel_step1_.setArg(idx++, *square_image);
  kernel_step1_.setArg(idx++, static_cast<int>(height));

  std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_step1_);
  std::string
      tuning_key = Concat("mvnorm_v_step1_opencl_kernel", gws[0], gws[1],
                          gws[2], normalize_variance_,
                          across_channels_);

  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_step1_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

// The second step of compute Variance Norm, read the (X - EX)^2 from
// square_image_ and compute (X - EX) / (E((X - EX)^2)^0.5 + eps_)
MaceStatus MVNormKernel::ExecuteVarianceNormStep2Kernel(
    OpContext *context, OpenCLRuntime *runtime, const uint32_t (&gws)[3],
    const Tensor *input, Tensor *output) {
  const auto height = input->dim(1);
  MACE_OUT_OF_RANGE_DEFINITION;
  if (kernel_step2_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    MACE_RETURN_IF_ERROR(BuildMVNKernel(runtime, &kernel_step2_,
                                        "mvnorm_vn_step2",
                                        &built_options, across_channels_));
    kwg_size_step2_ = static_cast<uint32_t>(
        runtime->GetKernelMaxWorkGroupSize(kernel_step2_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_step2_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_step2_);
  MACE_SET_3D_GWS_ARGS(kernel_step2_, gws);
  kernel_step2_.setArg(idx++, *(input->opencl_image()));
  cl::Image *mean_image = static_cast<cl::Image *>(mean_image_->buffer());
  kernel_step2_.setArg(idx++, *mean_image);
  cl::Image *square_image = static_cast<cl::Image *>(square_image_->buffer());
  kernel_step2_.setArg(idx++, *square_image);
  kernel_step2_.setArg(idx++, static_cast<int>(height));
  kernel_step2_.setArg(idx++, static_cast<float>(eps_));
  kernel_step2_.setArg(idx++, *(output->opencl_image()));

  std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_step2_);
  std::string
      tuning_key = Concat("mvnorm_v_step2_opencl_kernel", gws[0], gws[1],
                          gws[2], normalize_variance_,
                          across_channels_);

  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_step2_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
