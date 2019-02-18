// Copyright 2019 The MACE Authors. All Rights Reserved.
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


#include "mace/ops/ref/gemm.h"

namespace mace {
namespace ops {
namespace ref {

MaceStatus Gemm<float>::Compute(const OpContext *context,
                                const Tensor *lhs,
                                const Tensor *rhs,
                                const index_t batch,
                                const index_t rows,
                                const index_t cols,
                                const index_t depth,
                                const MatrixMajor lhs_major,
                                const MatrixMajor rhs_major,
                                const MatrixMajor output_major,
                                const bool lhs_batched,
                                const bool rhs_batched,
                                Tensor *output) {
  MACE_UNUSED(context);

  Tensor::MappingGuard lhs_guard(lhs);
  Tensor::MappingGuard rhs_guard(rhs);
  Tensor::MappingGuard output_guard(output);
  const float *lhs_data = lhs->data<float>();
  const float *rhs_data = rhs->data<float>();
  float *output_data = output->mutable_data<float>();

  for (index_t b = 0; b < batch; ++b) {
    MatrixMap<const float>
        lhs_matrix
        (lhs_data + static_cast<index_t>(lhs_batched) * b * rows * depth,
         lhs_major,
         rows,
         depth);
    MatrixMap<const float>
        rhs_matrix
        (rhs_data + static_cast<index_t>(rhs_batched) * b * depth * cols,
         rhs_major,
         depth,
         cols);
    MatrixMap<float>
        output_matrix(output_data + b * rows * cols, output_major, rows, cols);

    for (index_t r = 0; r < rows; ++r) {
      for (index_t c = 0; c < cols; ++c) {
        float sum = 0;
        for (index_t d = 0; d < depth; ++d) {
          sum += lhs_matrix(r, d) * rhs_matrix(d, c);
        }  // d

        *output_matrix.data(r, c) = sum;
      }  // c
    }  // r
  }   // b

  return MaceStatus::MACE_SUCCESS;
}

MaceStatus Gemm<float>::Compute(const OpContext *context,
                                const Tensor *lhs,
                                const Tensor *rhs,
                                const index_t batch,
                                const index_t lhs_rows,
                                const index_t lhs_cols,
                                const index_t rhs_rows,
                                const index_t rhs_cols,
                                const bool transpose_lhs,
                                const bool transpose_rhs,
                                const bool transpose_out,
                                const bool lhs_batched,
                                const bool rhs_batched,
                                Tensor *output) {
  index_t rows = transpose_lhs ? lhs_cols : lhs_rows;
  index_t depth = transpose_lhs ? lhs_rows : lhs_cols;
  index_t cols = transpose_rhs ? rhs_rows : rhs_cols;
  index_t depth2 = transpose_rhs ? rhs_cols : rhs_rows;
  MACE_CHECK(depth == depth2,
             "Matrices that multiply have inconsistent depth dim: ",
             depth,
             " vs. ",
             depth2);

  return Compute(context,
                 lhs,
                 rhs,
                 batch,
                 rows,
                 cols,
                 depth,
                 transpose_lhs ? ColMajor : RowMajor,
                 transpose_rhs ? ColMajor : RowMajor,
                 transpose_out ? ColMajor : RowMajor,
                 lhs_batched,
                 rhs_batched,
                 output);
}

}  // namespace ref
}  // namespace ops
}  // namespace mace
