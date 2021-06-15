// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/ops/opencl/buffer/transpose.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

MaceStatus TransposeKernel::Compute(OpContext *context,
                                    const Tensor *input,
                                    const std::vector<int> &dims,
                                    Tensor *output) {
  static const std::vector<int> nhwc2nchw_dims = {0, 3, 1, 2};
  static const std::vector<int> nchw2nhwc_dims = {0, 2, 3, 1};
  MACE_CHECK(dims == nhwc2nchw_dims || dims == nchw2nhwc_dims)
    << "Only suppport NHWC to NCHW or NCHW to NHWC";
  std::vector<index_t> output_shape = TransposeShape<index_t, index_t>(
      input->shape(), dims);
  MACE_RETURN_IF_ERROR(output->Resize(output_shape));
  const index_t d0 = input->dim(0);
  const index_t d1 = input->dim(1);
  const index_t d2 = input->dim(2);
  const index_t d3 = input->dim(3);
  const index_t d3_blk = RoundUpDiv4(d3);

  const uint32_t gws[3] = {static_cast<uint32_t>(d3_blk),
                           static_cast<uint32_t>(d2),
                           static_cast<uint32_t>(d1 * d0)};

  std::string kernel_name =
      (dims == nhwc2nchw_dims) ? "nhwc_to_nchw" : "nchw_to_nhwc";
  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;
  std::string cl_in_dt_str = DtToCLDt(input->dtype());
  std::string cl_out_dt_str = DtToCLDt(output->dtype());

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());

    built_options.emplace("-DIN_DATA_TYPE=" + cl_in_dt_str);
    built_options.emplace("-DOUT_DATA_TYPE=" + cl_out_dt_str);
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("transpose",
                                              obfuscated_kernel_name,
                                              built_options,
                                              &kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (IsResetArgsNeeded(context, input_shape_, input->shape())) {
    uint32_t idx = 0;
    MACE_BUFF_OUT_OF_RANGE_SET_ARGS(kernel_, output->size());
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_buffer()));
    kernel_.setArg(idx++, static_cast<int>(d1));
    kernel_.setArg(idx++, static_cast<int>(d3));
    kernel_.setArg(idx++, *(output->opencl_buffer()));
  }
  std::vector<uint32_t> lws = {4, 4, 4, 0};
  for (int i = 0; i < 3; ++i) {
    lws[i] = (lws[i] <= gws[i]) ? lws[i] : gws[i];
  }
  std::string tuning_key =
      Concat(kernel_name, output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace
