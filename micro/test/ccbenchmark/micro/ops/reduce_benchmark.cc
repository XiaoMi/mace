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
#include "micro/ops/reduce.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {
namespace {
template<typename T>
void Reduce(int32_t iters, const int32_t N,
            const int32_t H, const int32_t W, const int32_t C) {
  micro::testing::StopTiming();

  ReduceOp<T> reduce_op;
  framework::SubstituteOp substitude_op;
  const int32_t input_length = N * H * W * C;
  MACE_DEFINE_RANDOM_INPUT(T, input, input_length);
  T *output = common::test::GetGlobalBuffer()->GetBuffer<T>(input_length);
  int32_t input_dims[] = {N, H, W, C};
  int32_t output_dims[4] = {0};
  int32_t axis[] = {1, 2};
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("axis", axis, sizeof(axis) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);
  reduce_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);

  // Warm-up
  for (int32_t i = 0; i < 2; ++i) {
    reduce_op.Run();
  }

  micro::testing::StartTiming();
  while (iters--) {
    reduce_op.Run();
  }
}
}  // namespace

#define MICRO_BM_REDUCE_MACRO(N, H, W, C, TYPE)                      \
  static void MICRO_BM_REDUCE_##N##_##H##_##W##_##C##_##TYPE(        \
      int32_t iters) {                                               \
    const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C; \
    micro::testing::BytesProcessed(tot *(sizeof(TYPE)));             \
    Reduce<TYPE>(iters, N, H, W, C);                                 \
  }                                                                  \
  MICRO_BENCHMARK(MICRO_BM_REDUCE_##N##_##H##_##W##_##C##_##TYPE)

#define MICRO_BM_REDUCE(N, H, W, C) \
  MICRO_BM_REDUCE_MACRO(N, H, W, C, float)

MICRO_BM_REDUCE(1, 128, 128, 1);
MICRO_BM_REDUCE(4, 64, 64, 3);
MICRO_BM_REDUCE(2, 128, 128, 1);
MICRO_BM_REDUCE(2, 28, 28, 32);
MICRO_BM_REDUCE(1, 32, 32, 16);
MICRO_BM_REDUCE(1, 48, 64, 8);

}  // namespace test
}  // namespace ops
}  // namespace micro
