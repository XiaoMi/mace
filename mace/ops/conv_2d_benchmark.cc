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
static void Conv2d(int iters, int batch, int channels, int height, int width,
                   int kernel_h, int kernel_w, int stride,
                   Padding padding, int output_channels) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("Conv2d", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .Finalize(net.operator_def());

  // Add args
  net.AddIntsArg("strides", {stride, stride});
  net.AddIntArg("padding", padding);
  net.AddIntsArg("dilations", {1, 1});

  // Add input data
  net.AddRandomInput<float>("Input", {batch, channels, height, width});
  net.AddRandomInput<float>("Filter", {output_channels, channels, kernel_h, kernel_w});
  net.AddRandomInput<float>("Bias", {output_channels});

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }

  mace::testing::StartTiming();
  while(iters--) {
    net.RunOp(D);
  }
}

#define BM_CONV_2D_MACRO(N, C, H, W, KH, KW, STRIDE, P, OC, TYPE, DEVICE) \
  static void BM_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##_##P##_OC##_##TYPE##_##DEVICE(  \
        int iters) {                                                               \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;               \
    mace::testing::ItemsProcessed(tot);                                            \
    mace::testing::BytesProcessed(tot * (sizeof(TYPE)));                           \
    Conv2d<DEVICE, TYPE>(iters, N, C, H, W, KH, KW, STRIDE, mace::Padding::P, OC); \
  }                                                                                \
  BENCHMARK(BM_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##_##P##_OC##_##TYPE##_##DEVICE)

#define BM_CONV_2D(N, C, H, W, KH, KW, S, P, OC, TYPE)        \
  BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, OC, TYPE, CPU);  \
  BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, P, OC, TYPE, NEON);

BM_CONV_2D(1, 64, 32, 32, 1, 1, 1, VALID, 128, float);
BM_CONV_2D(1, 64, 32, 32, 3, 3, 1, VALID, 128, float);

} //  namespace mace
