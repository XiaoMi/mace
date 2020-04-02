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

#include "micro/ops/utils/gemm.h"

#include "micro/base/logging.h"

namespace micro {
namespace ops {

#ifndef MICRO_NOT_OPT
MaceStatus Gemm<mifloat>::Compute(const mifloat *lhs_data,
                                  const mifloat *rhs_data,
                                  const int32_t batch,
                                  const int32_t rows,
                                  const int32_t cols,
                                  const int32_t depth,
                                  const MatrixMajor lhs_major,
                                  const MatrixMajor rhs_major,
                                  const MatrixMajor output_major,
                                  const bool lhs_batched,
                                  const bool rhs_batched,
                                  mifloat *output_data) {
  for (int32_t b = 0; b < batch; ++b) {
    MatrixMap<const mifloat>
        lhs_matrix
        (lhs_data + static_cast<int32_t>(lhs_batched) * b * rows * depth,
         lhs_major,
         rows,
         depth);
    MatrixMap<const mifloat>
        rhs_matrix
        (rhs_data + static_cast<int32_t>(rhs_batched) * b * depth * cols,
         rhs_major,
         depth,
         cols);
    MatrixMap<mifloat>
        output_matrix(output_data + b * rows * cols, output_major, rows, cols);

    const int32_t rows_4 = rows / 4 * 4;
    const int32_t cols_4 = cols / 4 * 4;
    for (int32_t r = 0; r < rows; r += 4) {
      if (r < rows_4) {
        int32_t ro[4] = {r, r + 1, r + 2, r + 3};
        for (int32_t c = 0; c < cols; c += 4) {
          if (c < cols_4) {
            float sum[16] = {0};
            int32_t co[4] = {c, c + 1, c + 2, c + 3};
            for (int32_t d = 0; d < depth; ++d) {
              float lhs0 = lhs_matrix(ro[0], d);
              float lhs1 = lhs_matrix(ro[1], d);
              float lhs2 = lhs_matrix(ro[2], d);
              float lhs3 = lhs_matrix(ro[3], d);
              float rhs0 = rhs_matrix(d, co[0]);
              float rhs1 = rhs_matrix(d, co[1]);
              float rhs2 = rhs_matrix(d, co[2]);
              float rhs3 = rhs_matrix(d, co[3]);
              sum[0] += lhs0 * rhs0;
              sum[1] += lhs0 * rhs1;
              sum[2] += lhs0 * rhs2;
              sum[3] += lhs0 * rhs3;
              sum[4] += lhs1 * rhs0;
              sum[5] += lhs1 * rhs1;
              sum[6] += lhs1 * rhs2;
              sum[7] += lhs1 * rhs3;
              sum[8] += lhs2 * rhs0;
              sum[9] += lhs2 * rhs1;
              sum[10] += lhs2 * rhs2;
              sum[11] += lhs2 * rhs3;
              sum[12] += lhs3 * rhs0;
              sum[13] += lhs3 * rhs1;
              sum[14] += lhs3 * rhs2;
              sum[15] += lhs3 * rhs3;
            }  // d
            for (int32_t ro_i = 0; ro_i < 4; ++ro_i) {
              int32_t ro_i_base = ro_i * 4;
              for (int32_t co_i = 0; co_i < 4; ++co_i) {
                *output_matrix.data(ro[ro_i], co[co_i]) = sum[ro_i_base + co_i];
              }
            }
          } else {
            for (int32_t ro = r; ro < r + 4; ++ro) {
              for (int32_t co = cols_4; co < cols; ++co) {
                float sum = 0;
                for (int32_t d = 0; d < depth; ++d) {
                  sum += lhs_matrix(ro, d) * rhs_matrix(d, co);
                }  // d
                *output_matrix.data(ro, co) = sum;
              }
            }
          }
        }  // c
      } else {
        for (int32_t ro = rows_4; ro < rows; ++ro) {
          for (int32_t c = 0; c < cols; ++c) {
            float sum = 0;
            for (int32_t d = 0; d < depth; ++d) {
              sum += lhs_matrix(ro, d) * rhs_matrix(d, c);
            }  // d
            *output_matrix.data(ro, c) = sum;
          }  // c
        }
      }
    }  // r
  }   // b

  return MACE_SUCCESS;
}
#else
MaceStatus Gemm<mifloat>::Compute(const mifloat *lhs_data,
                                  const mifloat *rhs_data,
                                  const int32_t batch,
                                  const int32_t rows,
                                  const int32_t cols,
                                  const int32_t depth,
                                  const MatrixMajor lhs_major,
                                  const MatrixMajor rhs_major,
                                  const MatrixMajor output_major,
                                  const bool lhs_batched,
                                  const bool rhs_batched,
                                  mifloat *output_data) {
  for (int32_t b = 0; b < batch; ++b) {
    MatrixMap<const mifloat>
        lhs_matrix
        (lhs_data + static_cast<int32_t>(lhs_batched) * b * rows * depth,
         lhs_major,
         rows,
         depth);
    MatrixMap<const mifloat>
        rhs_matrix
        (rhs_data + static_cast<int32_t>(rhs_batched) * b * depth * cols,
         rhs_major,
         depth,
         cols);
    MatrixMap<mifloat>
        output_matrix(output_data + b * rows * cols, output_major, rows, cols);

    for (int32_t r = 0; r < rows; ++r) {
      for (int32_t c = 0; c < cols; ++c) {
        float sum = 0;
        for (int32_t d = 0; d < depth; ++d) {
          sum += lhs_matrix(r, d) * rhs_matrix(d, c);
        }  // d

        *output_matrix.data(r, c) = sum;
      }  // c
    }  // r
  }   // b

  return MACE_SUCCESS;
}
#endif

MaceStatus Gemm<mifloat>::Compute(const mifloat *lhs,
                                  const mifloat *rhs,
                                  const int32_t batch,
                                  const int32_t lhs_rows,
                                  const int32_t lhs_cols,
                                  const int32_t rhs_rows,
                                  const int32_t rhs_cols,
                                  const bool transpose_lhs,
                                  const bool transpose_rhs,
                                  const bool transpose_out,
                                  const bool lhs_batched,
                                  const bool rhs_batched,
                                  mifloat *output_data) {
  int32_t rows = transpose_lhs ? lhs_cols : lhs_rows;
  int32_t depth = transpose_lhs ? lhs_rows : lhs_cols;
  int32_t cols = transpose_rhs ? rhs_rows : rhs_cols;
  MACE_ASSERT1(depth == (transpose_rhs ? rhs_cols : rhs_rows),
               "Matrices that multiply have inconsistent depth dim: ");

  return Compute(lhs,
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
                 output_data);
}

}  // namespace ops
}  // namespace micro
