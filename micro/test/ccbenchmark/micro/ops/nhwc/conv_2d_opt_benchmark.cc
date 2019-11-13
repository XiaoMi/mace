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
#include "micro/ops/nhwc/conv_2d_c4_s4.h"
#include "micro/ops/substitute_op.h"
#include "micro/ops/test_utils.h"

namespace micro {
namespace ops {
namespace test {
namespace {
template<typename T>
void Conv2dOpt(int iters,
               const T *input, const int32_t *input_dims,
               const T *filter, const int32_t *filter_dims,
               const T *bias, T *output, int32_t *output_dims,
               int32_t stride, int32_t dilation, Padding padding) {
  micro::testing::StopTiming();

  Conv2dC4S4Op conv2d_opt_op;
  framework::SubstituteOp substitude_op;
  int32_t strides[] = {stride, stride};
  int32_t dilations[] = {dilation, dilation};
  substitude_op.AddInput(input, input_dims, 4)
      .AddInput(filter, filter_dims, 4)
      .AddInput(bias, filter_dims, 1)
      .AddRepeatArg("strides", strides, sizeof(strides) / sizeof(int32_t))
      .AddArg("padding", static_cast<int32_t>(padding))
      .AddRepeatArg("dilations", dilations, sizeof(dilations) / sizeof(int32_t))
      .AddOutput(output, output_dims, 4);
  conv2d_opt_op.Init(NULL, reinterpret_cast<framework::OpContext *>(
      &substitude_op), NULL);

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    conv2d_opt_op.Run();
  }

  micro::testing::StartTiming();
  while (iters--) {
    conv2d_opt_op.Run();
  }
}
}  // namespace

#define MICRO_BM_CONV_2D_OPT_MACRO(\
    N, H, W, C, KH, KW, STRIDE, DILATION, P, OC, TYPE)                       \
  static void                                                                \
    MICRO_BM_CONV_2D_OPT_##N##_##H##_##W##_##C##_K##KH##x##KW##S##STRIDE##D##\
        DILATION##_##P##_##OC##_##TYPE(int32_t iters) {                      \
    const int32_t input_length = N * H * W * C;                              \
    const int64_t tot = static_cast<int64_t>(iters) * input_length;          \
    int64_t pad_h = 0, pad_w = 0;                                            \
    if (P == SAME) {                                                         \
      pad_h = KH / 2;                                                        \
      pad_w = KW / 2;                                                        \
    }                                                                        \
    int64_t oh =                                                             \
        (H + 2 * pad_h - KH - (KH - 1) * (DILATION - 1)) / STRIDE + 1;       \
    int64_t ow =                                                             \
        (W + 2 * pad_w - KW - (KW - 1) * (DILATION - 1)) / STRIDE + 1;       \
    const int64_t macs = N * oh * ow * OC * KH * KW * C;                     \
    MACE_DEFINE_RANDOM_INPUT(TYPE, input, input_length);                     \
    const int32_t filter_length = OC * KH * KW * C;                          \
    MACE_DEFINE_RANDOM_INPUT(TYPE, filter, filter_length);                   \
    MACE_DEFINE_RANDOM_INPUT(TYPE, bias, (int32_t)OC);                       \
    const int32_t output_length = N * H * W * OC;                            \
    TYPE *output =                                                           \
        common::test::GetGlobalBuffer()->GetBuffer<TYPE>(output_length);     \
    int32_t input_dims[] = {N, H, W, C};                                     \
    int32_t filter_dims[] = {OC, KH, KW, C};                                 \
    int32_t output_dims[4] = {0};                                            \
    micro::testing::MacsProcessed(macs);                                     \
    micro::testing::BytesProcessed(tot *(sizeof(TYPE)));                     \
    Conv2dOpt<TYPE>(iters, input, input_dims,                                \
                         filter, filter_dims, bias, output,                  \
                         output_dims, STRIDE, DILATION, P);                  \
  }                                                                          \
  MICRO_BENCHMARK(                                                           \
      MICRO_BM_CONV_2D_OPT_##N##_##H##_##W##_##C##_K##KH##x##KW##S##STRIDE##\
        D##DILATION##_##P##_##OC##_##TYPE)

#define MICRO_BM_CONV_2D_OPT(N, H, W, C, KH, KW, S, D, P, OC) \
  MICRO_BM_CONV_2D_OPT_MACRO(N, H, W, C, KH, KW, S, D, P, OC, float)

MICRO_BM_CONV_2D_OPT(1, 32, 32, 64, 1, 1, 1, 1, VALID, 32);
MICRO_BM_CONV_2D_OPT(1, 33, 31, 64, 1, 1, 1, 1, VALID, 32);
MICRO_BM_CONV_2D_OPT(1, 32, 32, 64, 3, 3, 1, 1, SAME, 32);
MICRO_BM_CONV_2D_OPT(1, 33, 31, 64, 3, 3, 1, 1, SAME, 32);
MICRO_BM_CONV_2D_OPT(1, 32, 32, 64, 5, 5, 1, 1, SAME, 32);
MICRO_BM_CONV_2D_OPT(1, 32, 31, 64, 5, 5, 1, 1, SAME, 32);
MICRO_BM_CONV_2D_OPT(1, 32, 31, 64, 15, 1, 1, 1, SAME, 32);
MICRO_BM_CONV_2D_OPT(1, 32, 31, 64, 1, 15, 1, 1, SAME, 32);
MICRO_BM_CONV_2D_OPT(1, 32, 31, 64, 7, 7, 1, 1, SAME, 32);
MICRO_BM_CONV_2D_OPT(1, 32, 31, 64, 7, 7, 2, 1, SAME, 32);
MICRO_BM_CONV_2D_OPT(1, 32, 31, 64, 7, 7, 3, 1, SAME, 32);

}  // namespace test
}  // namespace ops
}  // namespace micro
