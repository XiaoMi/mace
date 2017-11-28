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
static void Conv2d(int iters,
                   int batch,
                   int channels,
                   int height,
                   int width,
                   int kernel_h,
                   int kernel_w,
                   int stride,
                   Padding padding,
                   int output_channels) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  net.AddRandomInput<D, float>("Filter",
                               {kernel_h, kernel_w, channels, output_channels});
  net.AddRandomInput<D, float>("Bias", {output_channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);
    BufferToImage<D>(net, "Filter", "FilterImage", kernels::BufferType::FILTER);
    BufferToImage<D>(net, "Bias", "BiasImage", kernels::BufferType::ARGUMENT);
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("Output")
        .AddIntsArg("strides", {stride, stride})
        .AddIntArg("padding", padding)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {stride, stride})
        .AddIntArg("padding", padding)
        .AddIntsArg("dilations", {1, 1})
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

// In common network, there are usually more than 1 layers, this is used to
// approximate the amortized latency. The OpenCL runtime for Mali/Adreno is
// in-order.

#define BM_CONV_2D_MACRO(N, C, H, W, KH, KW, STRIDE, P, OC, TYPE, DEVICE)                          \
  static void                                                                                      \
      BM_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##_##P##_##OC##_##TYPE##_##DEVICE( \
          int iters) {                                                                             \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;                               \
    mace::testing::ItemsProcessed(tot);                                                            \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                                            \
    Conv2d<DEVICE, TYPE>(iters, N, C, H, W, KH, KW, STRIDE, mace::Padding::P,                      \
                         OC);                                                                      \
  }                                                                                                \
  BENCHMARK(                                                                                       \
      BM_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##_##P##_##OC##_##TYPE##_##DEVICE)

#define BM_CONV_2D(N, C, H, W, KH, KW, S, P, OC, TYPE)        \
  BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, OC, TYPE, OPENCL);

// ICNet
BM_CONV_2D(1, 512, 15, 15, 1, 1, 1, VALID, 1024, float);
BM_CONV_2D(1, 128, 60, 60, 3, 3, 1, VALID, 128, float);
// SNPE GPU ExecutionDuration = 448us, % ALU Utilization = 105
BM_CONV_2D(1, 64, 60, 60, 1, 1, 1, VALID, 128, float);
// SNPE GPU ExecutionDuration = 258us, % ALU Utilization = 108
BM_CONV_2D(1, 32, 60, 60, 1, 1, 1, VALID, 128, float);

// SNPE GPU ExecutionDuration = 506us, % ALU Utilization = 106.8
BM_CONV_2D(1, 32, 60, 60, 3, 3, 1, VALID, 32, float);

// Test RGB <-> YUV
BM_CONV_2D(1, 3, 2160, 1080, 1, 1, 1, VALID, 3, float);
BM_CONV_2D(1, 3, 480, 480, 1, 1, 1, VALID, 3, float);

BM_CONV_2D(1, 64, 32, 32, 1, 1, 1, VALID, 128, float);
BM_CONV_2D(1, 64, 33, 31, 1, 1, 1, VALID, 128, float);  // Test bad alignments
BM_CONV_2D(1, 3, 512, 512, 1, 1, 1, VALID, 3, float);
BM_CONV_2D(1, 32, 112, 112, 1, 1, 1, VALID, 64, float);
BM_CONV_2D(1, 64, 56, 56, 1, 1, 1, VALID, 128, float);
BM_CONV_2D(1, 256, 28, 28, 1, 1, 1, VALID, 256, float);
BM_CONV_2D(1, 1024, 7, 7, 1, 1, 1, VALID, 1024, float);
BM_CONV_2D(1, 64, 32, 32, 3, 3, 1, VALID, 128, float);
BM_CONV_2D(1, 64, 33, 31, 3, 3, 1, VALID, 128, float);
BM_CONV_2D(1, 3, 512, 512, 3, 3, 1, VALID, 3, float);
BM_CONV_2D(1, 64, 32, 32, 3, 3, 1, SAME, 128, float);
BM_CONV_2D(1, 64, 33, 31, 3, 3, 1, SAME, 128, float);
BM_CONV_2D(1, 64, 32, 32, 3, 3, 2, VALID, 128, float);
BM_CONV_2D(1, 3, 512, 512, 3, 3, 2, VALID, 3, float);
BM_CONV_2D(1, 64, 33, 31, 3, 3, 2, VALID, 128, float);
BM_CONV_2D(1, 64, 32, 32, 3, 3, 2, SAME, 128, float);
BM_CONV_2D(1, 64, 33, 31, 3, 3, 2, SAME, 128, float);
BM_CONV_2D(1, 64, 32, 32, 5, 5, 1, VALID, 128, float);
BM_CONV_2D(1, 64, 32, 31, 5, 5, 1, VALID, 128, float);
BM_CONV_2D(1, 64, 32, 32, 5, 5, 1, SAME, 128, float);
BM_CONV_2D(1, 64, 32, 31, 5, 5, 1, SAME, 128, float);
}  //  namespace mace
