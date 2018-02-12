//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <algorithm>

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/conv_2d.h"
#include "mace/ops/ops_test_util.h"

namespace mace {

template <DeviceType D, typename T>
static void DepthwiseConv2d(int iters,
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
  net.AddRandomInput<D, float>("Input", {batch, height, width, input_channels});
  net.AddRandomInput<D, float>(
      "Filter", {kernel_h, kernel_w, input_channels, multiplier});
  net.AddRandomInput<D, float>("Bias", {input_channels * multiplier});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(net, "Filter", "FilterImage",
                        kernels::BufferType::DW_CONV2D_FILTER);
    BufferToImage<D, T>(net, "Bias", "BiasImage",
                        kernels::BufferType::ARGUMENT);
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("Output")
        .AddIntsArg("strides", {stride, stride})
        .AddIntArg("padding", padding)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
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
  }

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.RunOp(D);
    net.Sync();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
    net.Sync();
  }
}

#define BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, STRIDE, P, OC, TYPE,                                  \
                                   DEVICE)                                                                   \
  static void                                                                                                \
      BM_DEPTHWISE_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##_##P##_##OC##_##TYPE##_##DEVICE( \
          int iters) {                                                                                       \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;                                         \
    mace::testing::ItemsProcessed(tot);                                                                      \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                                                      \
    DepthwiseConv2d<DEVICE, TYPE>(iters, N, C, H, W, KH, KW, STRIDE,                                         \
                                  mace::Padding::P, OC);                                                     \
  }                                                                                                          \
  BENCHMARK(                                                                                                 \
      BM_DEPTHWISE_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##_##P##_##OC##_##TYPE##_##DEVICE)

#define BM_DEPTHWISE_CONV_2D(N, C, H, W, KH, KW, S, P, OC)                 \
  BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, OC, float, CPU);    \
  BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, OC, float, OPENCL); \
  BM_DEPTHWISE_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, OC, half, OPENCL);

BM_DEPTHWISE_CONV_2D(1, 32, 112, 112, 3, 3, 1, SAME, 1);
BM_DEPTHWISE_CONV_2D(1, 32, 112, 112, 3, 3, 2, SAME, 1);
BM_DEPTHWISE_CONV_2D(1, 64, 32, 32, 3, 3, 1, VALID, 1);
BM_DEPTHWISE_CONV_2D(1, 64, 33, 31, 3, 3, 1, VALID, 1);
BM_DEPTHWISE_CONV_2D(1, 64, 32, 32, 3, 3, 1, SAME, 1);
BM_DEPTHWISE_CONV_2D(1, 64, 33, 31, 3, 3, 1, SAME, 1);
BM_DEPTHWISE_CONV_2D(1, 3, 512, 512, 3, 3, 1, VALID, 1);
BM_DEPTHWISE_CONV_2D(1, 3, 512, 512, 3, 3, 1, SAME, 1);
BM_DEPTHWISE_CONV_2D(1, 64, 32, 32, 3, 3, 2, VALID, 1);
BM_DEPTHWISE_CONV_2D(1, 64, 33, 31, 3, 3, 2, VALID, 1);
BM_DEPTHWISE_CONV_2D(1, 64, 32, 32, 3, 3, 2, SAME, 1);
BM_DEPTHWISE_CONV_2D(1, 64, 33, 31, 3, 3, 2, SAME, 1);
BM_DEPTHWISE_CONV_2D(1, 3, 512, 512, 3, 3, 2, VALID, 1);
BM_DEPTHWISE_CONV_2D(1, 3, 512, 512, 3, 3, 2, SAME, 1);

}  // namespace mace
