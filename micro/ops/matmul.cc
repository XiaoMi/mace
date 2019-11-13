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

#include "micro/ops/matmul.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/scratch_buffer.h"
#include "micro/model/argument.h"

namespace micro {
namespace ops {

MaceStatus MatMulOp::OnInit() {
  transpose_a_ = GetArgByName("transpose_a", false);
  transpose_b_ = GetArgByName("transpose_b", false);
  input_a_ = GetInputData<mifloat>(INPUT_A);
  input_b_ = GetInputData<mifloat>(INPUT_B);
  bias_ = GetInputSize() > 3 ? GetInputData<mifloat>(BIAS) : NULL;
  output_ = GetOutputData<mifloat>(OUTPUT);

  input_a_dim_size_ = GetInputShapeDimSize(INPUT_A);
  input_b_dim_size_ = GetInputShapeDimSize(INPUT_B);

  input_a_dims_ = GetInputShapeDims(INPUT_A);
  input_b_dims_ = GetInputShapeDims(INPUT_B);

  MACE_ASSERT1(input_a_dim_size_ >= 2 && input_b_dim_size_ >= 2,
               "rank should be greater than or equal to 2");

  return MACE_SUCCESS;
}

MaceStatus MatMulOp::Run() {
  MACE_ASSERT(Validate());

  const int32_t lhs_rank = input_a_dim_size_;
  const int32_t lhs_rows = input_a_dims_[lhs_rank - 2];
  const int32_t lhs_cols = input_a_dims_[lhs_rank - 1];
  const int32_t rhs_rank = input_b_dim_size_;
  const int32_t rhs_rows = input_b_dims_[rhs_rank - 2];
  const int32_t rhs_cols = input_b_dims_[rhs_rank - 1];

  const int32_t rows = transpose_a_ ? lhs_cols : lhs_rows;
  const int32_t cols = transpose_b_ ? rhs_rows : rhs_cols;
  const int32_t depth = transpose_a_ ? lhs_rows : lhs_cols;
  const int32_t lhs_batch =
      base::accumulate_multi(input_a_dims_, 0, input_a_dim_size_ - 2);
  const int32_t rhs_batch =
      base::accumulate_multi(input_b_dims_, 0, input_b_dim_size_ - 2);
  int32_t *output_dims =
      ScratchBuffer(engine_config_).GetBuffer<int32_t>(input_a_dim_size_);

  int32_t batch = 1;
  base::memcpy(output_dims, input_a_dims_, input_a_dim_size_);
  if (lhs_rank >= rhs_rank) {
    output_dims[lhs_rank - 2] = rows;
    output_dims[lhs_rank - 1] = cols;
    batch = lhs_batch;
  } else {
    output_dims[rhs_rank - 2] = rows;
    output_dims[rhs_rank - 1] = cols;
    batch = rhs_batch;
  }
  bool lhs_batched = true;
  bool rhs_batched = true;
  if (lhs_rank < rhs_rank) {
    lhs_batched = false;
  } else if (rhs_rank < lhs_rank) {
    rhs_batched = false;
  }

  MACE_RETURN_IF_ERROR(
      ResizeOutputShape(OUTPUT, input_a_dim_size_, output_dims));

  if (rows == 1 && transpose_b_) {
    return gemv_.Compute(input_b_,
                         input_a_,
                         bias_,
                         batch,
                         cols,
                         depth,
                         rhs_batched,
                         lhs_batched,
                         output_);
  } else if (cols == 1 && !transpose_a_) {
    return gemv_.Compute(input_a_,
                         input_b_,
                         bias_,
                         batch,
                         rows,
                         depth,
                         lhs_batched,
                         rhs_batched,
                         output_);
  } else {
    MaceStatus ret = gemm_.Compute(input_a_,
                                   input_b_,
                                   batch,
                                   lhs_rows,
                                   lhs_cols,
                                   rhs_rows,
                                   rhs_cols,
                                   transpose_a_,
                                   transpose_b_,
                                   false,
                                   lhs_batched,
                                   rhs_batched,
                                   output_);
    if (bias_ != NULL) {
      MACE_ASSERT1(bias_dim_size_ == 1 && bias_dims_[0] == cols,
                   "bias' dim should be <= 2.");
      for (int32_t i = 0; i < batch * rows; ++i) {
        for (int32_t w = 0; w < cols; ++w) {
          int32_t idx = i * cols + w;
          output_[idx] = output_[idx] + bias_[w];
        }
      }
    }

    return ret;
  }
}

bool MatMulOp::Validate() {
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

  int32_t lhs_depth = transpose_a_ ? input_a_dims_[lhs_rank - 2] :
                      input_a_dims_[lhs_rank - 1];
  int32_t rhs_depth = transpose_b_ ? input_b_dims_[rhs_rank - 1] :
                      input_b_dims_[rhs_rank - 2];
  if (lhs_depth != rhs_depth) {
    MACE_ASSERT1(false, "the number of A's column must be equal to B's row ");
    return false;
  }

  return true;
}

}  // namespace ops
}  // namespace micro
