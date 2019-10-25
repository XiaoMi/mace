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

#include "mace/ops/opencl/image/lpnorm.h"

#include <set>
#include <string>
#include <vector>

namespace mace {
namespace ops {
namespace opencl {
namespace image {

LpNormKernel::LpNormKernel(const int p, const int axis) : p_(p), axis_(axis) {
  MACE_CHECK(p_ == 1 || p_ == 2, "Current p is: ", p);
}

MaceStatus LpNormKernel::Compute(OpContext *context,
                                 const Tensor *input, Tensor *output) {
  if (axis_ < 0) {
    axis_ += input->dim_size();
  }
  MACE_CHECK(axis_ == 1 || axis_ == 2 || axis_ == 3,
             "Current axis is: ", axis_);

  const auto batch = input->dim(0);
  const auto height = input->dim(1);
  const auto width = input->dim(2);
  const auto channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};
  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("lpnorm");
    built_options.emplace("-Dlpnorm=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    std::stringstream param_p;
    param_p << "-DPARAM_P=" << p_;
    built_options.emplace(param_p.str());
    std::stringstream param_axis;
    param_axis << "-DPARAM_AXIS=" << axis_;
    built_options.emplace(param_axis.str());
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("lpnorm", kernel_name,
                                              built_options, &kernel_));
    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_);
  uint32_t idx = 0;
  MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
  MACE_SET_3D_GWS_ARGS(kernel_, gws);
  kernel_.setArg(idx++, *(input->opencl_image()));
  kernel_.setArg(idx++, static_cast<int>(height));
  kernel_.setArg(idx++, static_cast<float>(1e-6));
  kernel_.setArg(idx++, *(output->opencl_image()));

  std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("lpnorm_opencl_kernel", batch, height, width, channels, p_, axis_);
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
