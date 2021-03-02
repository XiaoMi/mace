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

#include "mace/ops/opencl/image/space_to_batch.h"

#include "mace/runtimes/opencl/opencl_runtime.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus SpaceToBatchKernel::Compute(
    OpContext *context,
    const Tensor *space_tensor,
    const std::vector<int> &paddings,
    const std::vector<int> &block_shape,
    const std::vector<index_t> &output_shape,
    Tensor *batch_tensor) {
  MACE_RETURN_IF_ERROR(batch_tensor->Resize(output_shape));
  const char *kernel_name = "space_to_batch";
  const uint32_t chan_blk = RoundUpDiv4<uint32_t>(batch_tensor->dim(3));
  const uint32_t gws[3] = {
      chan_blk, static_cast<uint32_t>(batch_tensor->dim(2)),
      static_cast<uint32_t>(batch_tensor->dim(0) * batch_tensor->dim(1))};

  auto executor = OpenclRuntime::Get(context)->GetOpenclExecutor();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    auto input_dt = space_tensor->dtype();
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(input_dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(input_dt));

    MACE_RETURN_IF_ERROR(executor->BuildKernel("space_to_batch",
                                               obfuscated_kernel_name,
                                               built_options,
                                               &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(executor->GetKernelMaxWorkGroupSize(kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (IsResetArgsNeeded(context, input_shape_, space_tensor->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);

    kernel_.setArg(idx++, *(space_tensor->memory<cl::Image>()));
    kernel_.setArg(idx++, *(batch_tensor->memory<cl::Image>()));
    kernel_.setArg(idx++, block_shape[0]);
    kernel_.setArg(idx++, block_shape[1]);
    kernel_.setArg(idx++, paddings[0]);
    kernel_.setArg(idx++, paddings[2]);
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(0)));
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(2)));

    input_shape_ = space_tensor->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(executor, gws, kwg_size_);
  std::string tuning_key =
      Concat(kernel_name, batch_tensor->dim(0), batch_tensor->dim(1),
             batch_tensor->dim(2), batch_tensor->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(executor, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
