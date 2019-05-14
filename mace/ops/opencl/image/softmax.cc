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

#include "mace/ops/opencl/image/softmax.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

MaceStatus SoftmaxKernel::Compute(
    OpContext *context,
    const Tensor *logits,
    Tensor *output) {
  index_t batch = 0;
  index_t height = 0;
  index_t width = 0;
  index_t channels = 0;

  if (logits->dim_size() == 2) {
    batch = logits->dim(0);
    height = 1;
    width = 1;
    channels = logits->dim(1);

  } else if (logits->dim_size() == 4) {
    batch = logits->dim(0);
    height = logits->dim(1);
    width = logits->dim(2);
    channels = logits->dim(3);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  const index_t channel_blocks = RoundUpDiv4(channels);
  const int remain_channels = channel_blocks * 4 - channels;

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height * batch)};

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("softmax");
    built_options.emplace("-Dsoftmax=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(DT_FLOAT));
    if (use_log_)
      built_options.emplace("-DUSE_LOG");
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("softmax", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, logits->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(logits->opencl_image()));
    kernel_.setArg(idx++, static_cast<int>(channels));
    kernel_.setArg(idx++, remain_channels);
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = logits->shape();
  }

  std::vector<uint32_t> lws = softmax::LocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("softmax_opencl_kernel", batch, height, width, channels);
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
