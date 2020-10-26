// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "micro/ops/nhwc/cmsis_nn/arm_pooling_int8.h"

#include <arm_nnfunctions.h>

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/scratch_buffer.h"
#include "micro/include/utils/macros.h"
#include "micro/ops/nhwc/cmsis_nn/utilities.h"

namespace micro {
namespace ops {

void ArmPoolingInt8Op::MaxPooling(const mifloat *input,
                                  const int32_t *filter_hw,
                                  const int32_t *stride_hw,
                                  const int32_t *dilation_hw,
                                  const int32_t *pad_hw) {
  MACE_UNUSED(dilation_hw);

  cmsis_nn_context ctx;
  ctx.buf = NULL;
  ctx.size = 0;

  cmsis_nn_pool_params pool_params;
  pool_params.activation.min = -128;
  pool_params.activation.max = 127;
  pool_params.stride.h = stride_hw[0];
  pool_params.stride.w = stride_hw[1];
  pool_params.padding.h = pad_hw[0];
  pool_params.padding.w = pad_hw[1];

  MACE_ASSERT(input_dims_[0] == 1);

  cmsis_nn_dims input_dims;
  input_dims.n = input_dims_[0];
  input_dims.h = input_dims_[1];
  input_dims.w = input_dims_[2];
  input_dims.c = input_dims_[3];
  const int8_t *input_data = reinterpret_cast<const int8_t *>(input);

  cmsis_nn_dims filter_dims;
  filter_dims.h = filter_hw[0];
  filter_dims.w = filter_hw[1];

  cmsis_nn_dims output_dims;
  output_dims.n = output_dims_[0];
  output_dims.h = output_dims_[1];
  output_dims.w = output_dims_[2];
  output_dims.c = output_dims_[3];
  int8_t *output_data = reinterpret_cast<int8_t *>(output_);

  arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims,
                  &output_dims, output_data);
}

void ArmPoolingInt8Op::AvgPooling(const mifloat *input,
                                  const int32_t *filter_hw,
                                  const int32_t *stride_hw,
                                  const int32_t *dilation_hw,
                                  const int32_t *pad_hw) {
  MACE_UNUSED(dilation_hw);

  const int32_t out_width = output_dims_[2];
  const int32_t in_channels = input_dims_[3];

  cmsis_nn_context ctx;
  ctx.size = arm_avgpool_s8_get_buffer_size(out_width, in_channels);
  ScratchBuffer scratch_buffer(engine_config_);
  if (ctx.size > 0) {
    ctx.buf = scratch_buffer.GetBuffer<int8_t>(ctx.size);
  } else {
    ctx.buf = NULL;
  }

  cmsis_nn_pool_params pool_params;
  pool_params.activation.min = -128;
  pool_params.activation.max = 127;
  pool_params.stride.h = stride_hw[0];
  pool_params.stride.w = stride_hw[1];
  pool_params.padding.h = pad_hw[0];
  pool_params.padding.w = pad_hw[1];

  MACE_ASSERT(input_dims_[0] == 1);

  cmsis_nn_dims input_dims;
  input_dims.n = input_dims_[0];
  input_dims.h = input_dims_[1];
  input_dims.w = input_dims_[2];
  input_dims.c = input_dims_[3];
  const int8_t *input_data = reinterpret_cast<const int8_t *>(input);

  cmsis_nn_dims filter_dims;
  filter_dims.h = filter_hw[0];
  filter_dims.w = filter_hw[1];

  cmsis_nn_dims output_dims;
  output_dims.n = output_dims_[0];
  output_dims.h = output_dims_[1];
  output_dims.w = output_dims_[2];
  output_dims.c = output_dims_[3];
  int8_t *output_data = reinterpret_cast<int8_t *>(output_);

  arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims,
                 &output_dims, output_data);
}

}  // namespace ops
}  // namespace micro
