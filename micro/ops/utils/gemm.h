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

#ifndef MICRO_OPS_UTILS_GEMM_H_
#define MICRO_OPS_UTILS_GEMM_H_

#include "micro/base/types.h"
#include "micro/include/public/micro.h"
#include "micro/ops/utils/matrix.h"

namespace micro {
namespace ops {

template<typename T>
class Gemm {
 public:
  Gemm() {}
  ~Gemm() {}
  MaceStatus Compute(const mifloat *lhs_data,
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
                     T *output_data);
};

template<>
class Gemm<mifloat> {
 public:
  Gemm() {}
  ~Gemm() {}
  MaceStatus Compute(const mifloat *lhs_data,
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
                     mifloat *output_data);
  // Original matrix before transpose has row-major
  MaceStatus Compute(
      const mifloat *lhs_data,
      const mifloat *rhs_data,
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
      mifloat *output_data);
};

}  // namespace ops
}  // namespace micro


#endif  // MICRO_OPS_UTILS_GEMM_H_
