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
                          MeanType mean_type_) {
  std::stringstream macro_name;
  macro_name << "-Dmvnorm=" << kernel_name;
  built_options->emplace(macro_name.str());
  built_options->emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
  built_options->emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
  if (mean_type_ == MeanType::ACROSS_CHANNELS) {
    built_options->emplace("-DACROSS_CHANNELS");
  } else if (mean_type_ == MeanType::GROUP_CHANNELS) {
    built_options->emplace("-DGROUP_CHANNELS");
  }
  MACE_RETURN_IF_ERROR(runtime->BuildKernel("mvnorm", kernel_name,
                                            *built_options, kernel));
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace

MVNormKernel::MVNormKernel(bool normalize_variance,
                           MeanType mean_type, float eps, int group_num)
    : normalize_variance_(normalize_variance),
      mean_type_(mean_type),
      eps_(eps),
      group_num_(group_num) {}

MaceStatus MVNormKernel::Compute(OpContext *context,
                                 const Tensor *input, Tensor *output) {
  const auto batch = input->dim(0);
  const auto height = input->dim(1);
  const auto width = input->dim(2);
  const auto channels = input->dim(3);

  index_t group_blocks = 0;
  if (mean_type_ == MeanType::GROUP_CHANNELS) {
    MACE_CHECK(group_num_ > 0, "group num should > 0");
    const index_t group = channels / group_num_;
    MACE_CHECK(group > 0 && group % 4 == 0, group, " can not be divided by 4");
    group_blocks = group / 4;
  }

  return DoCompute(context, input, output, batch,
                   height, width, channels, group_blocks);
}

MaceStatus MVNormKernel::DoCompute(
    OpContext *context, const Tensor *input, Tensor *output,
    const index_t batch, const index_t height, const index_t width,
    const index_t channels, const index_t group_blocks) {
  const index_t channel_blocks = RoundUpDiv4(channels);
  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  auto runtime = context->device()->gpu_runtime()->opencl_runtime();

  const std::vector<index_t > mean_shape = {batch, 1, 1, channels};
  std::vector<size_t> mean_image_shape;
  OpenCLUtil::CalImage2DShape(mean_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &mean_image_shape);
  ScratchImageManager *scratch_manager =
      context->device()->gpu_runtime()->scratch_image_manager();
  ScratchImage scratch_mean_image(scratch_manager);
  auto mace_mean_image = scratch_mean_image.Scratch(
      context->device()->allocator(), mean_image_shape, input->dtype());
  cl::Image *mean_image = static_cast<cl::Image *>(mace_mean_image->buffer());

  if (normalize_variance_) {
    ScratchImage scratch_mean_image_sqr(scratch_manager);
    auto mace_mean_image_sqr = scratch_mean_image_sqr.Scratch(
        context->device()->allocator(), mean_image_shape, input->dtype());
    cl::Image *mean_image_sqr =
        static_cast<cl::Image *>(mace_mean_image_sqr->buffer());
    // compute the EX
    MACE_RETURN_IF_ERROR(ExecuteMeanValueKernel(
        context, runtime, batch, height, width, channel_blocks, group_blocks,
        input->opencl_image(), mean_image));
    // compute (X - EX)^2 to output
    MACE_RETURN_IF_ERROR(ExecuteVarianceNormStep1Kernel(
        context, runtime, gws, height, group_blocks, input->opencl_image(),
        mean_image, output->opencl_image()));
    // compute E((X - EX)^2) to mean_image_sqr_
    MACE_RETURN_IF_ERROR(ExecuteMeanValueKernel(
        context, runtime, batch, height, width, channel_blocks, group_blocks,
        output->opencl_image(), mean_image_sqr));
    // compute the compute (X - EX) / (E((X - EX)^2)^0.5 + eps_)
    MACE_RETURN_IF_ERROR(ExecuteVarianceNormStep2Kernel(
        context, runtime, gws, height, group_blocks, input->opencl_image(),
        mean_image, mean_image_sqr, output->opencl_image()));
  } else {
    // compute the EX
    MACE_RETURN_IF_ERROR(ExecuteMeanValueKernel(
        context, runtime, batch, height, width, channel_blocks, group_blocks,
        input->opencl_image(), mean_image));
    // compute the (X - EX)
    MACE_RETURN_IF_ERROR(ExecuteMeanNormKernel(
        context, runtime, gws, height, group_blocks, input->opencl_image(),
        mean_image, output->opencl_image()));
  }

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus MVNormKernel::ExecuteMeanValueKernel(OpContext *context,
                                                OpenCLRuntime *runtime,
                                                const index_t batch,
                                                const index_t height,
                                                const index_t width,
                                                const index_t channel_blocks,
                                                const index_t group_blocks,
                                                const cl::Image *input_image,
                                                cl::Image *output_image) {
  MACE_OUT_OF_RANGE_DEFINITION;
  if (kernel_mean_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    MACE_RETURN_IF_ERROR(
        BuildMVNKernel(runtime, &kernel_mean_, "mvnorm_compute_mean_value",
                       &built_options, mean_type_));
    kwg_size_mean_ = static_cast<uint32_t>(
        runtime->GetKernelMaxWorkGroupSize(kernel_mean_));
  }

  const uint32_t gws[2] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(batch)};
  const std::vector<uint32_t> lws = {static_cast<uint32_t>(kwg_size_mean_) / 8,
                                     8, 0};
  MACE_OUT_OF_RANGE_INIT(kernel_mean_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_mean_);
  MACE_SET_2D_GWS_ARGS(kernel_mean_, gws);
  kernel_mean_.setArg(idx++, *input_image);
  kernel_mean_.setArg(idx++, static_cast<int>(height));
  kernel_mean_.setArg(idx++, static_cast<int>(width));
  kernel_mean_.setArg(idx++, static_cast<int>(group_blocks));
  kernel_mean_.setArg(idx++, *output_image);

  std::string tuning_key = Concat(
      "mvnorm_compute_mean_opencl_kernel", gws[0],
      gws[1], normalize_variance_, mean_type_);

  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, kernel_mean_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus MVNormKernel::ExecuteMeanNormKernel(OpContext *context,
                                               OpenCLRuntime *runtime,
                                               const uint32_t (&gws)[3],
                                               const index_t height,
                                               const index_t group_blocks,
                                               const cl::Image *input,
                                               const cl::Image *mean_image,
                                               cl::Image *output) {
  MACE_OUT_OF_RANGE_DEFINITION;
  if (kernel_step1_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    MACE_RETURN_IF_ERROR(BuildMVNKernel(runtime, &kernel_step1_, "mvnorm_mean",
                                        &built_options, mean_type_));
    kwg_size_step1_ = static_cast<uint32_t>(
        runtime->GetKernelMaxWorkGroupSize(kernel_step1_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_step1_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_step1_);
  MACE_SET_3D_GWS_ARGS(kernel_step1_, gws);
  kernel_step1_.setArg(idx++, *input);
  kernel_step1_.setArg(idx++, *mean_image);
  kernel_step1_.setArg(idx++, static_cast<int>(height));
  kernel_step1_.setArg(idx++, static_cast<int>(group_blocks));
  kernel_step1_.setArg(idx++, *output);

  std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_step1_);
  std::string tuning_key = Concat("mvnorm_mean_opencl_kernel", gws[0], gws[1],
                                  gws[2], normalize_variance_,
                                  mean_type_, group_blocks);

  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_step1_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

// compute the (X - EX)^2
MaceStatus MVNormKernel::ExecuteVarianceNormStep1Kernel(
    OpContext *context, OpenCLRuntime *runtime,
    const uint32_t (&gws)[3], const index_t height, const index_t group_blocks,
    const cl::Image *input, const cl::Image *mean_image, cl::Image *output) {
  MACE_OUT_OF_RANGE_DEFINITION;
  if (kernel_step1_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    MACE_RETURN_IF_ERROR(BuildMVNKernel(runtime, &kernel_step1_,
                                        "mvnorm_vn_step1",
                                        &built_options, mean_type_));
    kwg_size_step1_ = static_cast<uint32_t>(
        runtime->GetKernelMaxWorkGroupSize(kernel_step1_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_step1_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_step1_);
  MACE_SET_3D_GWS_ARGS(kernel_step1_, gws);
  kernel_step1_.setArg(idx++, *input);
  kernel_step1_.setArg(idx++, *mean_image);
  kernel_step1_.setArg(idx++, *output);
  kernel_step1_.setArg(idx++, static_cast<int>(height));
  kernel_step1_.setArg(idx++, static_cast<int>(group_blocks));

  std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_step1_);
  std::string
      tuning_key = Concat("mvnorm_v_step1_opencl_kernel", gws[0], gws[1],
                          gws[2], normalize_variance_,
                          mean_type_);

  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_step1_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

// compute (X - EX) / (E((X - EX)^2)^0.5 + eps_)
MaceStatus MVNormKernel::ExecuteVarianceNormStep2Kernel(
    OpContext *context, OpenCLRuntime *runtime, const uint32_t (&gws)[3],
    const index_t height, const index_t group_blocks,
    const cl::Image *input, const cl::Image *mean_image,
    const cl::Image *mean_image_sqr, cl::Image *output) {
  MACE_OUT_OF_RANGE_DEFINITION;
  if (kernel_step2_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    MACE_RETURN_IF_ERROR(BuildMVNKernel(runtime, &kernel_step2_,
                                        "mvnorm_vn_step2",
                                        &built_options, mean_type_));
    kwg_size_step2_ = static_cast<uint32_t>(
        runtime->GetKernelMaxWorkGroupSize(kernel_step2_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_step2_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_step2_);
  MACE_SET_3D_GWS_ARGS(kernel_step2_, gws);
  kernel_step2_.setArg(idx++, *input);
  kernel_step2_.setArg(idx++, *mean_image);
  kernel_step2_.setArg(idx++, *mean_image_sqr);
  kernel_step2_.setArg(idx++, static_cast<int>(height));
  kernel_step2_.setArg(idx++, static_cast<int>(group_blocks));
  kernel_step2_.setArg(idx++, static_cast<float>(eps_));
  kernel_step2_.setArg(idx++, *output);

  std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_step2_);
  std::string
      tuning_key = Concat("mvnorm_v_step2_opencl_kernel", gws[0], gws[1],
                          gws[2], normalize_variance_,
                          mean_type_);

  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_step2_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
