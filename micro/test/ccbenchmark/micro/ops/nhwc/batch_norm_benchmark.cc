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
#include "micro/ops/nhwc/batch_norm.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {
namespace {
template<typename T>
void BatchNorm(int iters, const int N, const int H, const int W, const int C) {
  micro::testing::StopTiming();

  BatchNormOp batch_norm_op;
  framework::SubstituteOp substitude_op;
  const int32_t input_length = N * H * W * C;
  MACE_DEFINE_RANDOM_INPUT(T, input, input_length);
  MACE_DEFINE_RANDOM_INPUT(T, scale, static_cast<int32_t>(C));
  MACE_DEFINE_RANDOM_INPUT(T, offset, static_cast<int32_t>(C));
  MACE_DEFINE_RANDOM_INPUT(T, mean, static_cast<int32_t>(C));
  MACE_DEFINE_RANDOM_INPUT(T, var, static_cast<int32_t>(C));
  T *output = common::test::GetGlobalBuffer()->GetBuffer<T>(input_length);
  int32_t input_dims[] = {N, H, W, C};
  int32_t other_dims[] = {C};
  int32_t output_dims[4] = {0};
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(scale, other_dims, 1)
      .AddInput(offset, other_dims, 1)
      .AddInput(mean, other_dims, 1)
      .AddInput(var, other_dims, 1)
      .AddArg("epsilon", 1e-3)
      .AddOutput(output, output_dims, 4);
  batch_norm_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    batch_norm_op.Run();
  }

  micro::testing::StartTiming();
  while (iters--) {
    batch_norm_op.Run();
  }
}
}  // namespace

#define MICRO_BM_BATCH_NORM_MACRO(N, C, H, W, TYPE)                  \
  static void MICRO_BM_BATCH_NORM_##N##_##C##_##H##_##W##_##TYPE(    \
      int32_t iters) {                                               \
    const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C; \
    micro::testing::MacsProcessed(tot);                              \
    micro::testing::BytesProcessed(tot *(sizeof(TYPE)));             \
    BatchNorm<TYPE>(iters, N, H, W, C);                              \
  }                                                                  \
  MICRO_BENCHMARK(MICRO_BM_BATCH_NORM_##N##_##C##_##H##_##W##_##TYPE)

#define MICRO_BM_BATCH_NORM(N, C, H, W) \
  MICRO_BM_BATCH_NORM_MACRO(N, C, H, W, float);

MICRO_BM_BATCH_NORM(1, 128, 128, 1);
MICRO_BM_BATCH_NORM(1, 128, 128, 3);
MICRO_BM_BATCH_NORM(1, 64, 64, 3);
MICRO_BM_BATCH_NORM(1, 56, 56, 16);
MICRO_BM_BATCH_NORM(1, 28, 28, 64);
MICRO_BM_BATCH_NORM(1, 14, 14, 64);
MICRO_BM_BATCH_NORM(1, 14, 14, 32);
MICRO_BM_BATCH_NORM(1, 7, 7, 1024);

}  // namespace test
}  // namespace ops
}  // namespace micro
