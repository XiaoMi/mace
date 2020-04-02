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

#ifndef MICRO_OPS_MATMUL_H_
#define MICRO_OPS_MATMUL_H_

#include "micro/framework/operator.h"

#include "micro/ops/utils/gemv.h"
#include "micro/ops/utils/gemm.h"

namespace micro {
namespace ops {
class MatMulOp : public framework::Operator {
 public:
  MaceStatus OnInit();
  MaceStatus Run();

 private:
  bool Validate();

 private:
  const mifloat *input_a_;
  const int32_t *input_a_dims_;
  uint32_t input_a_dim_size_;

  const mifloat *input_b_;
  const int32_t *input_b_dims_;
  uint32_t input_b_dim_size_;

  const mifloat *bias_;
#ifndef NDEBUG
  const int32_t *bias_dims_;
  uint32_t bias_dim_size_;
#endif

  mifloat *output_;

  bool transpose_a_;
  bool transpose_b_;

  Gemv<mifloat> gemv_;
  Gemm<mifloat> gemm_;

  MACE_OP_INPUT_TAGS(INPUT_A, INPUT_B, BIAS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace micro

#endif  // MICRO_OPS_MATMUL_H_
