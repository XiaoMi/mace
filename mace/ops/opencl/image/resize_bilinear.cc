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

#include "mace/ops/opencl/image/resize_bilinear.h"

#include "mace/ops/common/utils.h"
#include "mace/runtimes/opencl/opencl_runtime.h"


namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus ResizeBilinearKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const index_t out_height,
    const index_t out_width,
    Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t in_height = input->dim(1);
  const index_t in_width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(out_width),
                           static_cast<uint32_t>(out_height * batch)};

  auto executor = OpenclRuntime::Get(context)->GetOpenclExecutor();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("resize_bilinear_nocache");
    built_options.emplace("-Dresize_bilinear_nocache=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    built_options.emplace(
        MakeString("-DCT_MODE=", coordinate_transformation_mode_));
    MACE_RETURN_IF_ERROR(
        executor->BuildKernel("resize_bilinear",
                              kernel_name,
                              built_options,
                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(executor->GetKernelMaxWorkGroupSize(kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (IsResetArgsNeeded(context, input_shape_, input->shape())) {
    MACE_CHECK(out_height > 0 && out_width > 0);
    std::vector<index_t> output_shape{batch, out_height, out_width, channels};
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    float height_scale =
        common::utils::CalculateResizeScale(in_height,
                                            out_height,
                                            align_corners_);
    float width_scale =
        common::utils::CalculateResizeScale(in_width,
                                            out_width,
                                            align_corners_);

    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    MACE_CHECK(input->memory_type() == MemoryType::GPU_IMAGE);
    MACE_CHECK(output->memory_type() == MemoryType::GPU_IMAGE);
    kernel_.setArg(idx++, *(input->memory<cl::Image>()));
    kernel_.setArg(idx++, *(output->mutable_memory<cl::Image>()));
    kernel_.setArg(idx++, height_scale);
    kernel_.setArg(idx++, width_scale);
    kernel_.setArg(idx++, static_cast<int32_t>(in_height));
    kernel_.setArg(idx++, static_cast<int32_t>(in_width));
    kernel_.setArg(idx++, static_cast<int32_t>(out_height));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t>
      lws = resize_bilinear::LocalWS(executor, gws, kwg_size_);
  std::string tuning_key =
      Concat("resize_bilinear_opencl_kernel", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(executor, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
