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

#include "mace/ops/opencl/image/depth_to_space.h"

#include "mace/runtimes/opencl/opencl_runtime.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {


MaceStatus DepthToSpaceKernel::Compute(OpContext *context,
                                       const Tensor *input,
                                       Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t input_height = input->dim(1);
  const index_t input_width = input->dim(2);
  const index_t input_depth = input->dim(3);

  MACE_CHECK(input_depth % (block_size_ * block_size_) == 0,
             "input depth should be dividable by block_size * block_size ",
             input_depth);

  const index_t output_height = input_height * block_size_;
  const index_t output_width = input_width * block_size_;
  const index_t output_depth = input_depth / (block_size_ * block_size_);
  MACE_CHECK(output_depth % 4 == 0 || output_depth < 4,
             "output channel not support:")
      << output_depth;

  std::vector<index_t> output_shape = {batch, output_height, output_width,
                                       output_depth};
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));

  uint32_t gws[3] = {0};
  if (mode_ == "DCR") {
    if (output_depth < 3) {
      gws[0] = static_cast<uint32_t>(RoundUpDiv4(input_depth));
      gws[1] = static_cast<uint32_t>(input_width);
      gws[2] = static_cast<uint32_t>(input_height * batch);
    } else {
      gws[0] = static_cast<uint32_t>(RoundUpDiv4(output_depth));
      gws[1] = static_cast<uint32_t>(output_width);
      gws[2] = static_cast<uint32_t>(output_height * batch);
    }
  } else {
    gws[0] = static_cast<uint32_t>(RoundUpDiv4(RoundUpDiv4(input_depth)));
    gws[1] = static_cast<uint32_t>(input_width);
    gws[2] = static_cast<uint32_t>(input_height * batch);
  }


  auto executor = OpenclRuntime::Get(context)->GetOpenclExecutor();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;

    const char *kernel_name = "depth_to_space";
    if (mode_ == "DCR") {
      if (output_depth < 4) {
        built_options.emplace(MakeString("-DDEPTH", output_depth));
        if (output_depth != 3) kernel_name = "depth_to_space_d1_d2";
      }
    } else {  // CRD
      MACE_CHECK(block_size_ == 2, "only blocksize_ == 2 is supported");
      kernel_name = "depth_to_space_crd_2x2";
    }

    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    auto dt = input->dtype();
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    MACE_RETURN_IF_ERROR(executor->BuildKernel(
        "depth_to_space", obfuscated_kernel_name, built_options, &kernel_));
    kwg_size_ =
        static_cast<uint32_t>(executor->GetKernelMaxWorkGroupSize(kernel_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (IsResetArgsNeeded(context, input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->memory<cl::Image>()));
    kernel_.setArg(idx++, static_cast<int32_t>(input_height));
    kernel_.setArg(idx++, static_cast<int32_t>(input_width));
    kernel_.setArg(idx++, static_cast<int32_t>(block_size_));
    kernel_.setArg(idx++, static_cast<int32_t>(output_height));
    kernel_.setArg(idx++, static_cast<int32_t>(output_width));
    kernel_.setArg(idx++, static_cast<int32_t>(output_depth));
    kernel_.setArg(idx++, *(output->mutable_memory<cl::Image>()));

    input_shape_ = input->shape();
  }

  std::string tuning_key = Concat("depth_to_space", batch, output_height,
                                  output_width, output_depth);
  const std::vector<uint32_t> lws = Default3DLocalWS(executor, gws, kwg_size_);
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(executor, kernel_, tuning_key, gws,
                                           lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
