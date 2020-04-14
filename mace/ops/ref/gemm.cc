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

#include "mace/ops/delegator/gemm.h"

namespace mace {
namespace ops {
namespace ref {

template<typename T>
class Gemm : public delegator::Gemm {
 public:
  explicit Gemm(const delegator::GemmParam &param) : delegator::Gemm(param) {}
  ~Gemm() {}
  MaceStatus Compute(const OpContext *context,
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
      Tensor *output) override;
};

template<typename T>
MaceStatus Gemm<T>::Compute(const OpContext *context,
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
  const T *lhs_data = lhs->data<T>();
  const T *rhs_data = rhs->data<T>();
  T *output_data = output->mutable_data<T>();

  for (index_t b = 0; b < batch; ++b) {
    MatrixMap<const T>
        lhs_matrix
        (lhs_data + static_cast<index_t>(lhs_batched) * b * rows * depth,
         lhs_major,
         rows,
         depth);
    MatrixMap<const T>
        rhs_matrix
        (rhs_data + static_cast<index_t>(rhs_batched) * b * depth * cols,
         rhs_major,
         depth,
         cols);
    MatrixMap<T>
        output_matrix(output_data + b * rows * cols, output_major, rows, cols);

    for (index_t r = 0; r < rows; ++r) {
      for (index_t c = 0; c < cols; ++c) {
        float sum = 0;
        for (index_t d = 0; d < depth; ++d) {
          sum += static_cast<float>(lhs_matrix(r, d)) *
              static_cast<float>(rhs_matrix(d, c));
        }  // d

        *output_matrix.data(r, c) = sum;
      }  // c
    }  // r
  }   // b

  return MaceStatus::MACE_SUCCESS;
}

template<typename T>
MaceStatus Gemm<T>::Compute(const OpContext *context,
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

  return Gemm<T>::Compute(context,
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

void RegisterGemmDelegator(OpDelegatorRegistry *registry) {
  MACE_REGISTER_DELEGATOR(
      registry, Gemm<float>, delegator::GemmParam,
      MACE_DELEGATOR_KEY(Gemm, DeviceType::CPU, float, ImplType::REF));
  MACE_REGISTER_BF16_DELEGATOR(
      registry, Gemm<BFloat16>, delegator::GemmParam,
      MACE_DELEGATOR_KEY(Gemm, DeviceType::CPU, BFloat16, ImplType::REF));
}

}  // namespace ref
}  // namespace ops
}  // namespace mace
