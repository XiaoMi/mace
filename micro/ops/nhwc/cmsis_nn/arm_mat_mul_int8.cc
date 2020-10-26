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

#include "micro/ops/nhwc/cmsis_nn/arm_mat_mul_int8.h"

#include <arm_nnfunctions.h>

#include "micro/base/logger.h"
#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/op_context.h"
#include "micro/framework/scratch_buffer.h"
#include "micro/model/argument.h"
#include "micro/model/const_tensor.h"
#include "micro/model/net_def.h"
#include "micro/ops/nhwc/cmsis_nn/utilities.h"

namespace micro {
namespace ops {

MaceStatus ArmMatMulInt8Op::OnInit() {
  transpose_a_ = GetArgByName("transpose_a", false);
  transpose_b_ = GetArgByName("transpose_b", false);
  input_a_ = GetInputData<int8_t>(INPUT_A);
  input_b_ = GetInputData<int8_t>(INPUT_B);
  output_ = GetOutputData<int8_t>(OUTPUT);

  if (GetInputSize() >= 3) {
    bias_ = GetInputData<int32_t>(BIAS);
    bias_dim_size_ = GetInputShapeDimSize(BIAS);
    bias_dims_ = GetInputShapeDims(BIAS);
  } else {
    bias_ = NULL;
    bias_dim_size_ = 0;
    bias_dims_ = NULL;
  }

  input_a_dim_size_ = GetInputShapeDimSize(INPUT_A);
  input_b_dim_size_ = GetInputShapeDimSize(INPUT_B);

  input_a_dims_ = GetInputShapeDims(INPUT_A);
  input_b_dims_ = GetInputShapeDims(INPUT_B);

  return MACE_SUCCESS;
}

MaceStatus ArmMatMulInt8Op::Run() {
  MACE_ASSERT(Validate());

  MACE_ASSERT(input_a_dim_size_ == 2);
  MACE_ASSERT(input_b_dim_size_ == 2);

  MACE_ASSERT(input_a_dims_[0] == 1);

  MACE_ASSERT(transpose_b_);
  MACE_ASSERT(!transpose_a_);

  const int32_t lhs_rows = input_a_dims_[0];
  const int32_t rhs_rows = input_b_dims_[0];
  const int32_t rhs_cols = input_b_dims_[1];

  const int32_t rhs_t_cols = rhs_rows;

  const int32_t rows = lhs_rows;
  const int32_t cols = rhs_t_cols;

  if (bias_ != NULL) {
    MACE_ASSERT(bias_dim_size_ == 1);
    MACE_ASSERT(bias_dims_[0] == cols);
  }

  int32_t *output_dims0 =
      ScratchBuffer(engine_config_).GetBuffer<int32_t>(input_a_dim_size_);

  output_dims0[0] = rows;
  output_dims0[1] = cols;

  MACE_RETURN_IF_ERROR(
      ResizeOutputShape(OUTPUT, input_a_dim_size_, output_dims0));

  QuantizeInfo input_quantize_info_a = GetInputQuantizeInfo(INPUT_A);
  QuantizeInfo input_quantize_info_b = GetInputQuantizeInfo(INPUT_B);
  QuantizeInfo output_quantize_info = GetOutputQuantizeInfo(OUTPUT);

  double double_multiplier = input_quantize_info_a.scale *
                             input_quantize_info_b.scale /
                             output_quantize_info.scale;
  int32_t multiplier;
  int32_t shift;
  QuantizeMultiplier(double_multiplier, &multiplier, &shift);

  ScratchBuffer scratch_buffer(engine_config_);

  int32_t *bias = NULL;
  if (bias_ == NULL) {
    bias = scratch_buffer.GetBuffer<int32_t>(cols);
    for (int32_t i = 0; i < cols; ++i) {
      bias[i] = 0;
    }
  } else {
    bias = const_cast<int32_t *>(bias_);
  }

  arm_status status = arm_nn_vec_mat_mult_t_s8(
      input_a_, input_b_, bias, output_, -input_quantize_info_a.zero,
      input_quantize_info_b.zero, output_quantize_info.zero, multiplier, shift,
      rhs_cols, rhs_rows, -128, 127);

  MACE_ASSERT(status == ARM_MATH_SUCCESS);

  return MACE_SUCCESS;
}

bool ArmMatMulInt8Op::Validate() {
  const int32_t lhs_rank = input_a_dim_size_;
  const int32_t rhs_rank = input_b_dim_size_;
  if (input_a_dim_size_ == input_b_dim_size_) {
    for (uint32_t i = 0; i < input_a_dim_size_ - 2; ++i) {
      MACE_ASSERT1(input_a_dims_[i] == input_b_dims_[i],
                   "batch dimensions are not equal");
    }
  } else {
    MACE_ASSERT1(input_a_dim_size_ == 2 || input_b_dim_size_ == 2,
                 "Either lhs or rhs matrix should has rank 2 "
                 "for non-batched matrix multiplication");
  }

  int32_t lhs_depth =
      transpose_a_ ? input_a_dims_[lhs_rank - 2] : input_a_dims_[lhs_rank - 1];
  int32_t rhs_depth =
      transpose_b_ ? input_b_dims_[rhs_rank - 1] : input_b_dims_[rhs_rank - 2];
  if (lhs_depth != rhs_depth) {
    MACE_ASSERT1(false, "the number of A's column must be equal to B's row ");
    return false;
  }

  return true;
}

}  // namespace ops
}  // namespace micro
