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

#include "mace/ops/opencl/buffer/utils.h"

#include <set>
#include <string>
#include <vector>

#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

MaceStatus PadInput(OpContext *context,
                    cl::Kernel *kernel,
                    const Tensor *input,
                    const int pad_top,
                    const int pad_left,
                    const bool input_changed,
                    Tensor *padded_input,
                    StatsFuture *future) {
  const index_t batch = input->dim(0);
  const index_t in_height = input->dim(1);
  const index_t in_width = input->dim(2);
  const index_t in_channel = input->dim(3);
  const index_t padded_height = padded_input->dim(1);
  const index_t padded_width = padded_input->dim(2);
  const index_t padded_channel = padded_input->dim(3);

  const uint32_t gws[2] = {
      static_cast<uint32_t>(padded_width * RoundUpDiv4(padded_channel)),
      static_cast<uint32_t>(padded_height * batch)
  };

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("Dpad_input");
    built_options.emplace("-Dpad_input=" + kernel_name);
    built_options.emplace("-DIN_DATA_TYPE=" + DtToCLDt(input->dtype()));
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(input->dtype()));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel(
        "buffer_transform",
        kernel_name,
        built_options,
        kernel));
  }

  MACE_OUT_OF_RANGE_INIT(*kernel);
  if (input_changed) {
    uint32_t idx = 0;
    MACE_BUFF_OUT_OF_RANGE_SET_ARGS(*kernel, padded_input->size());
    MACE_SET_2D_GWS_ARGS(*kernel, gws)
    kernel->setArg(idx++, *(input->opencl_buffer()));
    kernel->setArg(idx++, static_cast<int32_t>(in_height));
    kernel->setArg(idx++, static_cast<int32_t>(in_width));
    kernel->setArg(idx++, static_cast<int32_t>(in_channel));
    kernel->setArg(idx++, static_cast<int32_t>(padded_height));
    kernel->setArg(idx++, static_cast<int32_t>(padded_width));
    kernel->setArg(idx++, static_cast<int32_t>(padded_channel));
    kernel->setArg(idx++, pad_top);
    kernel->setArg(idx++, pad_left);
    kernel->setArg(idx++, *(padded_input->opencl_buffer()));
  }
  std::string tuning_key =
      Concat("pad_input", batch, in_height, in_width, in_channel,
             padded_height, padded_width, padded_channel);
  std::vector<uint32_t> lws = {8, 4, 0};
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, *kernel, tuning_key,
                                           gws, lws, future));
  MACE_OUT_OF_RANGE_VALIDATION
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace
