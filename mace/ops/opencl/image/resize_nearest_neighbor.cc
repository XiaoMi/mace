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

#include "mace/ops/opencl/image/resize_nearest_neighbor.h"

#include "mace/ops/common/utils.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus ResizeNearestNeighborKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const Tensor *size,
    const std::vector<index_t> &dims,
    Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t in_height = input->dim(1);
  const index_t in_width = input->dim(2);
  const index_t channels = input->dim(3);
  index_t out_height = 0;
  index_t out_width = 0;
  if (dims.size() < 2) {
    Tensor::MappingGuard size_mapper(size);
    out_height = size->data<int32_t>()[0];
    out_width = size->data<int32_t>()[1];
  } else {
    out_height = dims[0];
    out_width = dims[1];
  }
  const index_t channel_blocks = RoundUpDiv4(channels);

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(out_width),
                           static_cast<uint32_t>(out_height * batch)};

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL(
        "resize_nearest_neighbor_nocache");
    built_options.emplace("-Dresize_nearest_neighbor_nocache=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    MACE_RETURN_IF_ERROR(
        runtime->BuildKernel("resize_nearest_neighbor",
                             kernel_name,
                             built_options,
                             &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    MACE_CHECK(out_height > 0 && out_width > 0);
    std::vector<index_t> output_shape{batch, out_height, out_width, channels};

    std::vector<size_t> output_image_shape;
    OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                                &output_image_shape);
    MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

    float height_scale =
        common::utils::CalculateResizeScale(
            in_height, out_height, align_corners_);
    float width_scale =
        common::utils::CalculateResizeScale(
            in_width, out_width, align_corners_);

    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(output->opencl_image()));
    kernel_.setArg(idx++, height_scale);
    kernel_.setArg(idx++, width_scale);
    kernel_.setArg(idx++, static_cast<int32_t>(in_height));
    kernel_.setArg(idx++, static_cast<int32_t>(in_width));
    kernel_.setArg(idx++, static_cast<int32_t>(out_height));
    kernel_.setArg(idx++, static_cast<int32_t>(align_corners_));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t>
      lws = resize_nearest_neighbor::LocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("resize_nearest_neighbor_opencl_kernel", output->dim(0),
             output->dim(1), output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
