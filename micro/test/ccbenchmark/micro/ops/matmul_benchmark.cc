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

#include "micro/benchmark_utils/test_benchmark.h"
#include "micro/ops/matmul.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {
namespace {
template<typename T>
void MatMulBenchmark(int32_t iters, const int32_t N,
                     const int32_t H, const int32_t C, const int32_t OW) {
  micro::testing::StopTiming();

  MatMulOp matmul_op;
  framework::SubstituteOp substitude_op;
  const int32_t input0_length = N * H * C;
  MACE_DEFINE_RANDOM_INPUT(T, input0, input0_length);
  const int32_t input1_length = N * C * OW;
  MACE_DEFINE_RANDOM_INPUT(T, input1, input1_length);
  const int32_t output_length = N * H * OW;
  T *output = common::test::GetGlobalBuffer()->GetBuffer<T>(output_length);
  int32_t input0_dims[] = {N, H, C};
  int32_t input1_dims[] = {N, C, OW};
  int32_t output_dims[3] = {0};
  substitude_op.AddInput(input0, input0_dims, 3)
      .AddInput(input1, input1_dims, 3)
      .AddOutput(output, output_dims, 3);
  matmul_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);

  // Warm-up
  for (int32_t i = 0; i < 2; ++i) {
    matmul_op.Run();
  }

  micro::testing::StartTiming();
  while (iters--) {
    matmul_op.Run();
  }
}

template<typename T>
void MatMulTransposeBenchmark(int32_t iters, const int32_t N, const int32_t H,
                              const int32_t C, const int32_t OW) {
  micro::testing::StopTiming();

  MatMulOp matmul_op;
  framework::SubstituteOp substitude_op;
  const int32_t input0_length = N * H * C;
  MACE_DEFINE_RANDOM_INPUT(T, input0, input0_length);
  const int32_t input1_length = N * OW * C;
  MACE_DEFINE_RANDOM_INPUT(T, input1, input1_length);
  const int32_t output_length = N * H * OW;
  T *output = common::test::GetGlobalBuffer()->GetBuffer<T>(output_length);
  int32_t input0_dims[] = {N, H, C};
  int32_t input1_dims[] = {N, OW, C};
  int32_t output_dims[3] = {0};
  substitude_op.AddInput(input0, input0_dims, 3)
      .AddInput(input1, input1_dims, 3)
      .AddArg("transpose_b", 1)
      .AddOutput(output, output_dims, 3);
  matmul_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);

  // Warm-up
  for (int32_t i = 0; i < 2; ++i) {
    matmul_op.Run();
  }

  micro::testing::StartTiming();
  while (iters--) {
    matmul_op.Run();
  }
}

}  // namespace

#define MICRO_BM_MATMUL_MACRO(N, H, C, W, TYPE)                            \
  static void MICRO_BM_MATMUL_##N##_##H##_##C##_##W##_##TYPE(              \
      int32_t iters) {                                                     \
    const int64_t macs = N * H * W * C;                                    \
    const int64_t tot = static_cast<int64_t>(iters) * N * (C * H + H * W); \
    micro::testing::MacsProcessed(macs);                                   \
    micro::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    MatMulBenchmark<TYPE>(iters, N, H, C, W);                              \
  }                                                                        \
  MICRO_BENCHMARK(MICRO_BM_MATMUL_##N##_##H##_##C##_##W##_##TYPE)

#define MICRO_BM_MATMUL_OP(N, H, C, W) \
  MICRO_BM_MATMUL_MACRO(N, H, C, W, float)

MICRO_BM_MATMUL_OP(1, 300, 32, 1);
MICRO_BM_MATMUL_OP(1, 32, 64, 32);
MICRO_BM_MATMUL_OP(2, 16, 16, 49);
MICRO_BM_MATMUL_OP(3, 16, 16, 49);
MICRO_BM_MATMUL_OP(4, 16, 16, 49);
MICRO_BM_MATMUL_OP(4, 8, 32, 49);
MICRO_BM_MATMUL_OP(4, 32, 32, 49);

#define MICRO_BM_MATMUL_TRANSPOSE_MACRO(N, H, C, W, TYPE)                  \
  static void MICRO_BM_MATMUL_##T_##N##_##H##_##C##_##W##_##TYPE(          \
      int32_t iters) {                                                     \
    const int64_t macs = N * H * W * C;                                    \
    const int64_t tot = static_cast<int64_t>(iters) * N * (C * H + H * W); \
    micro::testing::MacsProcessed(macs);                                   \
    micro::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    MatMulBenchmark<TYPE>(iters, N, H, C, W);                              \
  }                                                                        \
  MICRO_BENCHMARK(MICRO_BM_MATMUL_##T_##N##_##H##_##C##_##W##_##TYPE)

#define MICRO_BM_MATMUL_TRANSPOSE(N, H, C, W) \
  MICRO_BM_MATMUL_TRANSPOSE_MACRO(N, H, C, W, float)

MICRO_BM_MATMUL_TRANSPOSE(4, 8, 32, 49);
MICRO_BM_MATMUL_TRANSPOSE(2, 16, 16, 49);

}  // namespace test
}  // namespace ops
}  // namespace micro
