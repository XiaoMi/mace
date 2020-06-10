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

#ifndef MACE_OPS_ARM_BASE_GEMM_H_
#define MACE_OPS_ARM_BASE_GEMM_H_

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/common/matrix.h"
#include "mace/ops/delegator/gemm.h"
#include "mace/public/mace.h"
#include "mace/utils/math.h"

// This implements matrix-matrix multiplication.
// In the case of matrix-vector multiplication, use gemv.h/gemv.cc instead

namespace mace {
namespace ops {
namespace arm {

enum { kNoCache, kCacheLhs, kCacheRhs };

template<typename T>
class Gemm : public delegator::Gemm {
 public:
  explicit Gemm(const delegator::GemmParam &param)
      : delegator::Gemm(param), pack_cache_(GetCPUAllocator()),
        should_cache_pack_(param.should_cache_pack_),
        cached_(0) {}
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
      Tensor *output) override;

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
      Tensor *output) override {
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

 protected:
  void ComputeBlock(const T *packed_lhs_data,
                    const T *packed_rhs_data,
                    const index_t depth_padded,
                    T *packed_output_data);

  void PackLhs(const MatrixMap<const T> &lhs,
               T *packed_lhs);

  void PackRhs(const MatrixMap<const T> &rhs,
               T *packed_rhs);

  void UnpackOutput(const T *packed_output,
                    MatrixMap<T> *output);
  template<int RowBlockSize, int ColBlockSize>
  void Unpack(const T *packed_output,
              MatrixMap<T> *output) {
    const index_t rows = output->rows();
    const index_t cols = output->cols();
    for (index_t r = 0; r < rows; ++r) {
      for (index_t c = 0; c < cols; ++c) {
        *output->data(r, c) = packed_output[r * ColBlockSize + c];
      }
    }
  }

  template<int WidthBlockSize, int DepthBlockSize>
  void Pack(const MatrixMap<const T> &matrix,
            MatrixMajor dst_major,
            T *packed_matrix) {
    const index_t rows = matrix.rows();
    const index_t cols = matrix.cols();
    index_t depth = cols;
    if (dst_major == RowMajor) {
      // rhs
      depth = rows;
    }
    const index_t depth_padded = RoundUp(depth, static_cast<index_t>(4));
    memset(static_cast<void *>(packed_matrix), 0,
           sizeof(T) * WidthBlockSize * depth_padded);
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

 private:
  Buffer pack_cache_;
  bool should_cache_pack_;
  int cached_;
};

}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_GEMM_H_
