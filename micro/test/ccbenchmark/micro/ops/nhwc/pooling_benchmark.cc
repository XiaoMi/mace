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
#include "micro/ops/nhwc/pooling_ref.h"
#include "micro/ops/nhwc/pooling_s4.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {
namespace {
template<typename T>
void Pooling(int iters, const T *input, const int32_t *input_dims,
             T *output, int32_t *output_dims, int32_t kernel,
             int32_t stride, Padding padding, PoolingType pooling_type) {
  micro::testing::StopTiming();

  PoolingS4Op pooling_op;
  framework::SubstituteOp substitude_op;
  int32_t strides[] = {stride, stride};
  int32_t kernels[] = {kernel, kernel};
  int32_t dilations[] = {1, 1};
  substitude_op.AddInput(input, input_dims, 4)
      .AddArg("pooling_type", static_cast<int32_t>(pooling_type))
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", static_cast<int32_t>(padding))
      .AddRepeatArg("kernels", kernels, sizeof(kernels) / sizeof(int32_t))
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);
  pooling_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    pooling_op.Run();
  }

  micro::testing::StartTiming();
  while (iters--) {
    pooling_op.Run();
  }
}
}  // namespace

#define MICRO_BM_POOLING_MACRO(N, H, W, C, KE, STRIDE, PA, PO, TYPE)     \
  static void                                                            \
      MICRO_BM_POOLING_##N##_##H##_##W##_##C##_K##KE##S##STRIDE##_##PA##_\
        ##PO##_##TYPE(int32_t iters) {                                   \
    const int32_t input_length = N * H * W * C;                          \
    const int64_t tot = static_cast<int64_t>(iters) * input_length;      \
    micro::testing::BytesProcessed(tot *(sizeof(TYPE)));                 \
    MACE_DEFINE_RANDOM_INPUT(TYPE, input, input_length);                 \
    const int32_t output_length = input_length;                          \
    TYPE *output =                                                       \
        common::test::GetGlobalBuffer()->GetBuffer<TYPE>(output_length); \
    int32_t input_dims[] = {N, H, W, C};                                 \
    int32_t output_dims[4] = {0};                                        \
    Pooling<TYPE>(iters, input, input_dims,                              \
                  output, output_dims, KE, STRIDE, PA, PO);              \
  }                                                                      \
  MICRO_BENCHMARK(                                                       \
      MICRO_BM_POOLING_##N##_##H##_##W##_##C##_K##KE##S##STRIDE##_##PA##_\
        ##PO##_##TYPE)

#define MICRO_BM_POOLING(N, H, W, C, K, S, PA, PO) \
  MICRO_BM_POOLING_MACRO(N, H, W, C, K, S, PA, PO, float)

MICRO_BM_POOLING(1, 129, 129, 3, 2, 2, SAME, MAX);
MICRO_BM_POOLING(1, 65, 65, 3, 2, 2, SAME, MAX);
MICRO_BM_POOLING(1, 48, 64, 8, 48, 64, VALID, AVG);
MICRO_BM_POOLING(1, 7, 7, 8, 7, 1, VALID, AVG);

}  // namespace test
}  // namespace ops
}  // namespace micro
