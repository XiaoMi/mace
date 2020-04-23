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
#include "micro/ops/activation.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {

namespace {
void ActivationBenchmark(const char *activation_type, int iters,
                         const float *input, const int32_t *input_dims,
                         float *output, int32_t *output_dims) {
  micro::testing::StopTiming();

  const uint32_t arg_type_len = base::strlen(activation_type);
  ActivationOp activation_op;
  framework::SubstituteOp substitude_op;
  substitude_op.AddInput(input, input_dims, 4)
      .AddRepeatArg("activation", activation_type, arg_type_len)
      .AddOutput(output, output_dims, 4);
  MACE_DEFINE_RANDOM_INPUT(float, alpha, input_dims[3]);
  if (base::strcmp(activation_type, "PRELU") == 0) {
    substitude_op.AddInput(alpha, input_dims + 3, 1);
  }
  activation_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    activation_op.Run();
  }

  micro::testing::StartTiming();
  while (iters--) {
    activation_op.Run();
  }
}
}  // namespace

#define MICRO_BM_ACTIVATION_MACRO(N, H, W, C, TYPE)                      \
  static void MICRO_BM##_##TYPE##_##N##_##H##_##W##_##C(int32_t iters) { \
    const int32_t buffer_length = N * H * W * C;                         \
    MACE_DEFINE_RANDOM_INPUT(float, input, buffer_length);               \
    float *output =                                                      \
        common::test::GetGlobalBuffer()->GetBuffer<float>(buffer_length);\
    int32_t input_dims[] = {N, H, W, C};                                 \
    int32_t output_dims[4] = {0};                                        \
    const int64_t tot = static_cast<int64_t>(iters) * buffer_length;     \
    micro::testing::BytesProcessed(tot *(sizeof(float)));                \
    ActivationBenchmark(#TYPE, iters, input,                             \
                        input_dims, output, output_dims);                \
  }                                                                      \
  MICRO_BENCHMARK(MICRO_BM##_##TYPE##_##N##_##H##_##W##_##C)

#define MICRO_BM_RELU(N, H, W, C) \
  MICRO_BM_ACTIVATION_MACRO(N, H, W, C, RELU)

MICRO_BM_RELU(1, 4, 4, 1);
MICRO_BM_RELU(1, 128, 128, 1);

#define MICRO_BM_RELUX(N, H, W, C) \
  MICRO_BM_ACTIVATION_MACRO(N, H, W, C, RELUX)

MICRO_BM_RELUX(1, 4, 4, 1);
MICRO_BM_RELUX(1, 128, 128, 1);

#define MICRO_BM_PRELU(N, H, W, C) \
  MICRO_BM_ACTIVATION_MACRO(N, H, W, C, PRELU)

MICRO_BM_PRELU(1, 4, 4, 1);
MICRO_BM_PRELU(1, 128, 128, 1);

#define MICRO_BM_TANH(N, H, W, C) \
  MICRO_BM_ACTIVATION_MACRO(N, H, W, C, TANH)

MICRO_BM_TANH(1, 4, 4, 1);
MICRO_BM_TANH(1, 128, 128, 1);

#define MICRO_BM_SIGMOID(N, H, W, C) \
  MICRO_BM_ACTIVATION_MACRO(N, H, W, C, SIGMOID)

MICRO_BM_SIGMOID(1, 4, 4, 1);
MICRO_BM_SIGMOID(1, 128, 128, 1);

}  // namespace test
}  // namespace ops
}  // namespace micro
