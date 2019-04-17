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

#ifndef MACE_OPS_ARM_FP32_GEMM_H_
#define MACE_OPS_ARM_FP32_GEMM_H_

#include "mace/public/mace.h"
#include "mace/core/tensor.h"
#include "mace/core/op_context.h"
#include "mace/ops/common/matrix.h"
#include "mace/utils/math.h"

// This implements matrix-matrix multiplication.
// In the case of matrix-vector multiplication, use gemv.h/gemv.cc instead

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

class Gemm {
 public:
  explicit Gemm(const bool should_cache_pack)
      : pack_cache_(GetCPUAllocator()),
        should_cache_pack_(should_cache_pack),
        cached_(0) {}
  Gemm() : Gemm(false) {}
  ~Gemm() {}

  MaceStatus Compute(
      const OpContext *context,
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
      Tensor *output);

  // Original matrix before transpose has row-major
  MaceStatus Compute(
      const OpContext *context,
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
      Tensor *output);

 private:
  void ComputeBlock(const float *packed_lhs_data,
                    const float *packed_rhs_data,
                    const index_t depth_padded,
                    float *packed_output_data);

  void PackLhs(const MatrixMap<const float> &lhs,
               float *packed_lhs);

  void PackRhs(const MatrixMap<const float> &rhs,
               float *packed_rhs);

  void UnpackOutput(const float *packed_output,
                    MatrixMap<float> *output);

  template<int RowBlockSize, int ColBlockSize>
  void Unpack(const float *packed_output,
              MatrixMap<float> *output) {
    const index_t rows = output->rows();
    const index_t cols = output->cols();
    for (index_t r = 0; r < rows; ++r) {
      for (index_t c = 0; c < cols; ++c) {
        *output->data(r, c) = packed_output[r * ColBlockSize + c];
      }
    }
  }

  template<int WidthBlockSize, int DepthBlockSize>
  void Pack(const MatrixMap<const float> &matrix,
            MatrixMajor dst_major,
            float *packed_matrix) {
    const index_t rows = matrix.rows();
    const index_t cols = matrix.cols();
    index_t depth = cols;
    if (dst_major == RowMajor) {
      // rhs
      depth = rows;
    }
    const index_t depth_padded = RoundUp(depth, static_cast<index_t>(4));
    memset(packed_matrix, 0, sizeof(float) * WidthBlockSize * depth_padded);
    if (dst_major == ColMajor) {
      for (index_t c = 0; c < cols; ++c) {
        for (index_t r = 0; r < rows; ++r) {
          packed_matrix[c * WidthBlockSize + r] = matrix(r, c);
        }
      }
    } else {
      for (index_t r = 0; r < rows; ++r) {
        for (index_t c = 0; c < cols; ++c) {
          packed_matrix[r * WidthBlockSize + c] = matrix(r, c);
        }
      }
    }
  }

  Buffer pack_cache_;

  bool should_cache_pack_;
  int cached_;
};

template<>
void Gemm::Pack<4, 4>(const MatrixMap<const float> &matrix,
                      MatrixMajor dst_major,
                      float *packed_matrix);

template<>
void Gemm::Pack<8, 4>(const MatrixMap<const float> &matrix,
                      MatrixMajor dst_major,
                      float *packed_matrix);

template<>
void Gemm::Unpack<4, 8>(const float *packed_output, MatrixMap<float> *output);

template<>
void Gemm::Unpack<8, 8>(const float *packed_output, MatrixMap<float> *output);

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_FP32_GEMM_H_
