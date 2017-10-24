//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/channel_shuffle.h"
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;
using namespace mace::kernels;

template <DeviceType D>
static void ChannelShuffle(
    int iters, int batch, int channels, int height, int width, int group) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("GlobalAvgPooling", "GlobalAvgPoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  net.AddIntArg("group", group);
  net.AddRandomInput<DeviceType::CPU, float>("Input", {batch, channels, height, width});

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
}

#define BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, DEVICE)                  \
  static void BM_CHANNEL_SHUFFLE_##N##_##C##_##H##_##W##_##G##_##DEVICE( \
      int iters) {                                                       \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;     \
    mace::testing::ItemsProcessed(tot);                                  \
    mace::testing::BytesProcessed(tot *(sizeof(float)));                 \
    ChannelShuffle<DEVICE>(iters, N, C, H, W, G);                        \
  }                                                                      \
  BENCHMARK(BM_CHANNEL_SHUFFLE_##N##_##C##_##H##_##W##_##G##_##DEVICE)

#define BM_CHANNEL_SHUFFLE(N, C, H, W, G) \
  BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, CPU);

BM_CHANNEL_SHUFFLE(1, 64, 64, 64, 8);
BM_CHANNEL_SHUFFLE(1, 64, 128, 128, 8);
BM_CHANNEL_SHUFFLE(1, 64, 256, 256, 8);
