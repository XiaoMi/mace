// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_OPS_MATMUL_H_
#define MACE_OPS_MATMUL_H_

#include "mace/core/operator.h"
#include "mace/kernels/matmul.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class MatMulOp : public Operator<D, T> {
 public:
  MatMulOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        transpose_a_(OperatorBase::GetOptionalArg<bool>("transpose_a", false)),
        transpose_b_(OperatorBase::GetOptionalArg<bool>("transpose_b", false)) {
  }

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *A = this->Input(INPUT_A);
    const Tensor *B = this->Input(INPUT_B);
    Tensor *C = this->Output(OUTPUT);
    MACE_CHECK(A->dim_size() == B->dim_size() && A->dim_size() >= 2,
               "rank(A) should be equal to rank(B), rank should be greater "
               "than or equal to 2");
    index_t rank = A->dim_size();
    for (index_t i = 0; i < rank - 2; ++i) {
      MACE_CHECK(A->dim(i) == B->dim(i), "batch dimensions are not equal");
    }
    index_t ak = transpose_a_ ? A->dim(rank - 2) : A->dim(rank - 1);
    index_t bk = transpose_b_ ? B->dim(rank - 1) : B->dim(rank - 2);
    MACE_CHECK(ak == bk, "the number of A's column ", ak,
               " must be equal to B's row ", bk);

    return functor_(A, B, C, transpose_a_, transpose_b_, future);
  }

 private:
  MACE_OP_INPUT_TAGS(INPUT_A, INPUT_B);
  MACE_OP_OUTPUT_TAGS(OUTPUT);

  kernels::MatMulFunctor<D, T> functor_;
  bool transpose_a_;
  bool transpose_b_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_MATMUL_H_
