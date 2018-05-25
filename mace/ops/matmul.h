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
      : Operator<D, T>(operator_def, ws) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *A = this->Input(0);
    const Tensor *B = this->Input(1);
    Tensor *C = this->Output(0);
    MACE_CHECK(A->dim_size() == 4 && 4 == B->dim_size())
        << "The dimension of A and B should be 4";
    MACE_CHECK(A->dim(0) == B->dim(0)) << "A and B must have same batch size";
    MACE_CHECK(A->dim(2) == B->dim(1))
        << "the number of A's column " << A->dim(2)
        << " must be equal to B's row " << B->dim(1);

    return functor_(A, B, C, future);
  }

 private:
  kernels::MatMulFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_MATMUL_H_
