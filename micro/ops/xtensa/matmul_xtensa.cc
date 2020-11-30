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

#include "micro/ops/xtensa/matmul_xtensa.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/scratch_buffer.h"
#include "micro/model/argument.h"
#include "nnlib/xa_nnlib_api.h"

namespace micro {
namespace ops {

MaceStatus MatMulXtensaOp::OnInit() {
  transpose_a_ = GetArgByName("transpose_a", false);
  transpose_b_ = GetArgByName("transpose_b", false);
  input_a_ = GetInputData<mifloat>(INPUT_A);
  input_b_ = GetInputData<mifloat>(INPUT_B);
  output_ = GetOutputData<mifloat>(OUTPUT);

  bias_ = NULL;
  if (GetInputSize() >= 3) {
    bias_ = GetInputData<mifloat>(BIAS);
    bias_dim_size_ = GetInputShapeDimSize(BIAS);
    bias_dims_ = GetInputShapeDims(BIAS);
  }

  input_a_dim_size_ = GetInputShapeDimSize(INPUT_A);
  input_b_dim_size_ = GetInputShapeDimSize(INPUT_B);

  input_a_dims_ = GetInputShapeDims(INPUT_A);
  input_b_dims_ = GetInputShapeDims(INPUT_B);

  MACE_ASSERT1(input_a_dim_size_ >= 2 && input_b_dim_size_ >= 2,
               "rank should be greater than or equal to 2");

  return MACE_SUCCESS;
}

MaceStatus MatMulXtensaOp::Run() {
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

  ScratchBuffer scratch_buffer(engine_config_);

  float *bias = NULL;
  if (bias_ == NULL) {
    bias = scratch_buffer.GetBuffer<float>(cols);
    for (int32_t i = 0; i < cols; ++i) {
      bias[i] = 0;
    }
  } else {
    bias = const_cast<float *>(bias_);
  }

  int re = xa_nn_matXvec_f32xf32_f32(output_, input_b_, NULL, input_a_, NULL,
                                     bias_, rhs_rows, rhs_cols, 0, rhs_cols, 0);
  MACE_ASSERT(re == 0);

  return MACE_SUCCESS;
}

bool MatMulXtensaOp::Validate() {
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
