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


#ifndef MACE_OPS_DELEGATOR_GEMM_H_
#define MACE_OPS_DELEGATOR_GEMM_H_

#include "mace/core/ops/op_context.h"
#include "mace/core/ops/op_delegator.h"
#include "mace/core/registry/op_delegator_registry.h"
#include "mace/ops/common/matrix.h"

namespace mace {
namespace ops {
namespace delegator {

struct GemmParam : public DelegatorParam {
  explicit GemmParam(const bool should_cache_pack = false)
      : should_cache_pack_(should_cache_pack) {}

  const bool should_cache_pack_;
};

class Gemm : public OpDelegator {
 public:
  explicit Gemm(const GemmParam &param) : OpDelegator(param) {}
  virtual ~Gemm() = default;

  MACE_DEFINE_DELEGATOR_CREATOR(Gemm)

  virtual MaceStatus Compute(const OpContext *context,
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
                             Tensor *output) = 0;
  // Original matrix before transpose has row-major
  virtual MaceStatus Compute(const OpContext *context,
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
                             Tensor *output) = 0;
};

}  // namespace delegator
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DELEGATOR_GEMM_H_

