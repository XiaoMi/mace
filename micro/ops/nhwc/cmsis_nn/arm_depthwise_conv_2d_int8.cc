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

#include "micro/ops/nhwc/cmsis_nn/arm_depthwise_conv_2d_int8.h"

#include <arm_nnfunctions.h>

#include "micro/base/logger.h"
#include "micro/framework/op_context.h"
#include "micro/framework/scratch_buffer.h"
#include "micro/model/const_tensor.h"
#include "micro/model/net_def.h"
#include "micro/ops/nhwc/cmsis_nn/utilities.h"

namespace micro {
namespace ops {

MaceStatus ArmDepthwiseConv2dInt8Op::Compute(int32_t (&output_dims)[4]) {
  QuantizeInfo input_quantize_info = GetInputQuantizeInfo(INPUT);
  QuantizeInfo filter_quantize_info = GetInputQuantizeInfo(FILTER);
  QuantizeInfo output_quantize_info = GetOutputQuantizeInfo(OUTPUT);

  double double_multiplier = input_quantize_info.scale *
                             filter_quantize_info.scale /
                             output_quantize_info.scale;
  int32_t multiplier;
  int32_t shift;
  QuantizeMultiplier(double_multiplier, &multiplier, &shift);

  cmsis_nn_dw_conv_params dw_conv_params;
  dw_conv_params.ch_mult = filter_dims_[0];
  /// input_offset is negative
  dw_conv_params.input_offset = -input_quantize_info.zero;
  dw_conv_params.output_offset = output_quantize_info.zero;
  dw_conv_params.activation.min = -128;
  dw_conv_params.activation.max = 127;
  dw_conv_params.stride.w = strides_[1];
  dw_conv_params.stride.h = strides_[0];
  dw_conv_params.padding.w = padding_sizes_[1] / 2;
  dw_conv_params.padding.h = padding_sizes_[0] / 2;
  dw_conv_params.dilation.w = dilations_[1];
  dw_conv_params.dilation.h = dilations_[0];

  ScratchBuffer scratch_buffer(engine_config_);

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = scratch_buffer.GetBuffer<int32_t>(output_dims[3]);
  quant_params.shift = scratch_buffer.GetBuffer<int32_t>(output_dims[3]);
  for (int32_t i = 0; i < output_dims[3]; ++i) {
    quant_params.multiplier[i] = multiplier;
    quant_params.shift[i] = shift;
  }

  MACE_ASSERT(input_dims_[0] == 1);
  MACE_ASSERT(filter_dims_[0] == 1);
  MACE_ASSERT(dilations_[0] == 1 && dilations_[1] == 1);

  cmsis_nn_dims input_dims;
  input_dims.n = input_dims_[0];
  input_dims.h = input_dims_[1];
  input_dims.w = input_dims_[2];
  input_dims.c = input_dims_[3];
  const int8_t *input_data = reinterpret_cast<const int8_t *>(input_);

  cmsis_nn_dims filter_dims;
  filter_dims.n = filter_dims_[0];
  filter_dims.h = filter_dims_[1];
  filter_dims.w = filter_dims_[2];
  filter_dims.c = filter_dims_[3];
  const int8_t *filter_data = reinterpret_cast<const int8_t *>(filter_);

  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output_dims[3];
  int32_t *bias_data =
      const_cast<int32_t *>(reinterpret_cast<const int32_t *>(bias_));
  if (bias_data == NULL) {
    bias_data = scratch_buffer.GetBuffer<int32_t>(output_dims[3]);
    for (int32_t i = 0; i < bias_dims.c; ++i) {
      bias_data[i] = 0;
    }
  }

  cmsis_nn_dims cmn_output_dims;
  cmn_output_dims.n = output_dims[0];
  cmn_output_dims.h = output_dims[1];
  cmn_output_dims.w = output_dims[2];
  cmn_output_dims.c = filter_dims.c * filter_dims.n;
  int8_t *output_data = reinterpret_cast<int8_t *>(output_);

  cmsis_nn_context cmn_context;
  cmn_context.size = arm_depthwise_conv_wrapper_s8_get_buffer_size(
      &dw_conv_params, &input_dims, &filter_dims, &cmn_output_dims);

  if (cmn_context.size > 0) {
    cmn_context.buf = scratch_buffer.GetBuffer<int8_t>(cmn_context.size);
  } else {
    cmn_context.buf = NULL;
  }

  arm_status status = arm_depthwise_conv_wrapper_s8(
      &cmn_context, &dw_conv_params, &quant_params, &input_dims, input_data,
      &filter_dims, filter_data, &bias_dims, bias_data, &cmn_output_dims,
      output_data);
  MACE_ASSERT(status == ARM_MATH_SUCCESS)
      << "failed in arm_convolve_wrapper_s8";

  return MACE_SUCCESS;
}

MaceStatus ArmDepthwiseConv2dInt8Op::Run() {
  int32_t output_dims[4] = {0};
  InitPaddingAndOutputSize(input_dims_, filter_dims_, FLOOR, output_dims);
  output_dims[3] *= input_dims_[3];
  ResizeOutputShape(0, 4, output_dims);

  MACE_RETURN_IF_ERROR(Compute(output_dims));

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
