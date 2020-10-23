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

#include "micro/ops/nhwc/cmsis_nn/arm_eltwise_int8.h"

#include <arm_nnfunctions.h>

#include "micro/base/logging.h"
#include "micro/base/types.h"
#include "micro/base/utils.h"
#include "micro/ops/nhwc/cmsis_nn/utilities.h"

namespace micro {
namespace ops {

MaceStatus ArmEltwiseInt8Op::OnInit() {
  MACE_ASSERT(GetInputSize() == 2);

  input0_ = GetInputData<int8_t>(INPUT0);
  input0_dims_ = GetInputShapeDims(INPUT0);
  input0_dim_size_ = GetInputShapeDimSize(INPUT0);

  input1_ = GetInputData<int8_t>(INPUT1);
  input1_dims_ = GetInputShapeDims(INPUT1);
  input1_dim_size_ = GetInputShapeDimSize(INPUT1);

  output_ = GetOutputData<int8_t>(OUTPUT);

  type_ = static_cast<eltwise::Type>(
      GetArgByName("type", static_cast<int32_t>(NONE)));
  coeff_ = GetRepeatArgByName<float>("coeff", &coeff_size_);

  return MACE_SUCCESS;
}

MaceStatus ArmEltwiseInt8Op::Run() {
  MACE_ASSERT1(GetInputSize() == 2,
               "ArmEltwiseInt8Op only supports 2 inputs");
  MACE_ASSERT(input0_dim_size_ == input1_dim_size_);
  MACE_ASSERT(base::ShapeIsEqual(input0_dims_, input1_dims_, input1_dim_size_));

  MACE_RETURN_IF_ERROR(
          ResizeOutputShape(OUTPUT, input0_dim_size_, input0_dims_));

  if (type_ == eltwise::SUM) {
    QuantizeInfo input_quantize_info0 = GetInputQuantizeInfo(0);
    QuantizeInfo input_quantize_info1 = GetInputQuantizeInfo(1);
    QuantizeInfo output_quantize_info = GetOutputQuantizeInfo(OUTPUT);

    int32_t input0_offset = -input_quantize_info0.zero;
    double input0_scale = input_quantize_info0.scale;
    int32_t input1_offset = -input_quantize_info1.zero;
    double input1_scale = input_quantize_info1.scale;
    int32_t output_offset = output_quantize_info.zero;
    double output_scale = output_quantize_info.scale;

    int32_t left_shift = 20;

    const double twice_max_input_scale =
        2 * static_cast<double>(base::max(input0_scale, input1_scale));
    const double real_input0_multiplier =
        static_cast<double>(input0_scale) / twice_max_input_scale;
    const double real_input1_multiplier =
        static_cast<double>(input1_scale) / twice_max_input_scale;
    const double real_output_multiplier =
        twice_max_input_scale /
        ((1 << left_shift) * static_cast<double>(output_scale));

    int32_t input0_multiplier = 0;
    int32_t input0_shift = 0;
    QuantizeMultiplier(real_input0_multiplier, &input0_multiplier,
                       &input0_shift);

    int32_t input1_multiplier = 0;
    int32_t input1_shift = 0;
    QuantizeMultiplier(real_input1_multiplier, &input1_multiplier,
                       &input1_shift);

    int32_t output_multiplier = 0;
    int32_t output_shift = 0;
    QuantizeMultiplier(real_output_multiplier, &output_multiplier,
                       &output_shift);

    int32_t element_size = base::GetShapeSize(input0_dim_size_, input0_dims_);
    arm_elementwise_add_s8(input0_, input1_, input0_offset, input0_multiplier,
                           input0_shift, input1_offset, input1_multiplier,
                           input1_shift, left_shift, output_, output_offset,
                           output_multiplier, output_shift, -128, 127,
                           element_size);
  } else {
    MACE_ASSERT1(false, "Unsupported ArmEltwiseInt8Op type");
  }

  return MACE_SUCCESS;
}

}  // namespace ops
}  // namespace micro
