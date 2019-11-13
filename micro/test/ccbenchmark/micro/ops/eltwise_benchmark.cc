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
#include "micro/ops/eltwise.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {
namespace {
template<typename T>
void EltwiseBenchmark(int32_t iters, eltwise::Type type, const int32_t N,
                      const int32_t H, const int32_t W, const int32_t C) {
  micro::testing::StopTiming();

  EltwiseOp<T> eltwise_op;
  framework::SubstituteOp substitude_op;
  const int32_t input_length = N * H * W * C;
  MACE_DEFINE_RANDOM_INPUT(T, input0, input_length);
  MACE_DEFINE_RANDOM_INPUT(T, input1, input_length);
  T *output = common::test::GetGlobalBuffer()->GetBuffer<T>(input_length);
  int32_t input_dims[] = {N, H, W, C};
  int32_t output_dims[4] = {0};
  T coeffs[] = {1.2, 2.1};
  substitude_op.AddInput(input0, input_dims, 4)
      .AddInput(input1, input_dims, 4)
      .AddArg("type", static_cast<int32_t>(type))
      .AddRepeatArg("coeff", coeffs, sizeof(coeffs) / sizeof(T))
      .AddOutput(output, output_dims, 4);
  eltwise_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);

  // Warm-up
  for (int32_t i = 0; i < 2; ++i) {
    eltwise_op.Run();
  }

  micro::testing::StartTiming();
  while (iters--) {
    eltwise_op.Run();
  }
}
}  // namespace

#define MICRO_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, TYPE)           \
  static void                                                        \
      MICRO_BM_ELTWISE_##ELT_TYPE##_##N##_##H##_##W##_##C##_##TYPE(  \
          int32_t iters) {                                           \
    const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C; \
    micro::testing::BytesProcessed(tot *(sizeof(TYPE)));             \
    EltwiseBenchmark<TYPE>(                                          \
        iters, static_cast<eltwise::Type>(ELT_TYPE), N, H, W, C);    \
  }                                                                  \
  MICRO_BENCHMARK(                                                   \
      MICRO_BM_ELTWISE_##ELT_TYPE##_##N##_##H##_##W##_##C##_##TYPE)

#define MICRO_BM_ELTWISE(ELT_TYPE, N, H, W, C) \
  MICRO_BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, float)

MICRO_BM_ELTWISE(2, 1, 32, 32, 8);
MICRO_BM_ELTWISE(2, 1, 60, 60, 16);
MICRO_BM_ELTWISE(2, 1, 64, 64, 8);
MICRO_BM_ELTWISE(0, 1, 32, 32, 8);
MICRO_BM_ELTWISE(0, 1, 60, 60, 16);
MICRO_BM_ELTWISE(5, 1, 32, 32, 8);
MICRO_BM_ELTWISE(5, 1, 60, 60, 16);

}  // namespace test
}  // namespace ops
}  // namespace micro
