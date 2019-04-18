// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include <algorithm>

#include "mace/utils/statistics.h"
#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void DepthwiseConv2d(int iters,
                     int batch,
                     int input_channels,
                     int height,
                     int width,
                     int kernel_h,
                     int kernel_w,
                     int stride,
                     Padding padding,
                     int multiplier) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    if (DataTypeToEnum<T>::value != DT_UINT8) {
      net.AddRandomInput<D, float>(
          "Input", {batch, input_channels, height, width});
    } else {
      net.AddRandomInput<DeviceType::CPU, uint8_t>(
          "Input", {batch, height, width, input_channels});
      net.GetTensor("Input")->SetScale(0.1);
    }

  } else if (D == DeviceType::GPU) {
    net.AddRandomInput<D, float>(
        "Input", {batch, height, width, input_channels});
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  if (DataTypeToEnum<T>::value != DT_UINT8) {
    net.AddRandomInput<D, float>(
        "Filter", {multiplier, input_channels, kernel_h, kernel_w}, true);
    net.AddRandomInput<D, float>("Bias", {input_channels * multiplier}, true);
  } else {
    net.AddRandomInput<DeviceType::CPU, uint8_t>(
        "Filter", {kernel_h, kernel_w, input_channels, multiplier}, true);
    net.GetTensor("Filter")->SetScale(0.1);
    net.AddRandomInput<DeviceType::CPU, int32_t>(
        "Bias", {input_channels * multiplier}, true);
  }

  OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {stride, stride})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  net.Setup(D);

  if (DataTypeToEnum<T>::value == DT_UINT8) {
    net.GetTensor("Output")->SetScale(0.1);
  }

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.Run();
    net.Sync();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
    net.Sync();
  }
}
}  // namespace

#define MACE_BM_DEPTHWISE_CONV_2D_MACRO(                                       \
    N, C, H, W, KH, KW, STRIDE, P, M, TYPE, DEVICE)                            \
  static void                                                                  \
      MACE_BM_DEPTHWISE_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE\
        ##_##P##_##M##_##TYPE##_##DEVICE(                                      \
          int iters) {                                                         \
    const int64_t dilation = 1;                                                \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;           \
    int64_t pad_h = 0, pad_w = 0;                                              \
    if (P == SAME) {                                                           \
      pad_h = KH / 2;                                                          \
      pad_w = KW / 2;                                                          \
    }                                                                          \
    int64_t oh =                                                               \
        (H + 2 * pad_h - KH - (KH - 1) * (dilation - 1)) / STRIDE + 1;         \
    int64_t ow =                                                               \
        (W + 2 * pad_w - KW - (KW - 1) * (dilation - 1)) / STRIDE + 1;         \
    const int64_t macs =                                                       \
        static_cast<int64_t>(iters) * mace::benchmark::StatMACs(               \
            "DepthwiseConv2d", {M, C, KH, KW}, {N, oh, ow, C});                \
    mace::testing::MacsProcessed(macs);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    DepthwiseConv2d<DEVICE, TYPE>(iters, N, C, H, W, KH, KW, STRIDE,           \
                                  mace::Padding::P, M);                        \
  }                                                                            \
  MACE_BENCHMARK(                                                              \
      MACE_BM_DEPTHWISE_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE\
        ##_##P##_##M##_##TYPE##_##DEVICE)

#if defined(MACE_ENABLE_OPENCL) && defined(MACE_ENABLE_QUANTIZE)
#define MACE_BM_DEPTHWISE_CONV_2D(N, C, H, W, KH, KW, S, P, M)                 \
  MACE_BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, M, float, CPU);    \
  MACE_BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, M, float, GPU);    \
  MACE_BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, M, half, GPU);     \
  MACE_BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, M, uint8_t, CPU)
#elif defined(MACE_ENABLE_OPENCL)
#define MACE_BM_DEPTHWISE_CONV_2D(N, C, H, W, KH, KW, S, P, M)                 \
  MACE_BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, M, float, CPU);    \
  MACE_BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, M, float, GPU);    \
  MACE_BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, M, half, GPU)
#elif defined(MACE_ENABLE_QUANTIZE)
#define MACE_BM_DEPTHWISE_CONV_2D(N, C, H, W, KH, KW, S, P, M)                 \
  MACE_BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, M, float, CPU);    \
  MACE_BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, M, uint8_t, CPU)
#else
#define MACE_BM_DEPTHWISE_CONV_2D(N, C, H, W, KH, KW, S, P, M)                 \
  MACE_BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, M, float, CPU)
#endif

MACE_BM_DEPTHWISE_CONV_2D(1, 32, 112, 112, 3, 3, 1, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 32, 56, 56, 3, 3, 2, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 32, 112, 112, 3, 3, 2, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 32, 224, 224, 3, 3, 2, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 56, 56, 3, 3, 2, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 112, 112, 3, 3, 2, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 224, 224, 3, 3, 2, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 32, 32, 3, 3, 1, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 33, 31, 3, 3, 1, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 32, 32, 3, 3, 1, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 33, 31, 3, 3, 1, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 3, 512, 512, 3, 3, 1, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 3, 512, 512, 3, 3, 1, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 32, 32, 3, 3, 2, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 33, 31, 3, 3, 2, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 32, 32, 3, 3, 2, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 64, 33, 31, 3, 3, 2, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 3, 512, 512, 3, 3, 2, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 3, 512, 512, 3, 3, 2, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 3, 112, 112, 3, 3, 2, VALID, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 3, 224, 224, 3, 3, 2, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 8, 224, 224, 3, 3, 2, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 128, 56, 56, 3, 3, 1, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 128, 56, 56, 3, 3, 2, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 256, 28, 28, 3, 3, 1, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 256, 28, 28, 3, 3, 2, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 512, 14, 14, 3, 3, 1, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 512, 14, 14, 3, 3, 2, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 1024, 7, 7, 3, 3, 1, SAME, 1);
MACE_BM_DEPTHWISE_CONV_2D(1, 1024, 7, 7, 3, 3, 2, SAME, 1);

}  // namespace test
}  // namespace ops
}  // namespace mace
