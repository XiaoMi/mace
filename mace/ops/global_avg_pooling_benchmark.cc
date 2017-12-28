//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/global_avg_pooling.h"
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;
using namespace mace::kernels;

template <DeviceType D>
static void GlobalAvgPooling(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("GlobalAvgPooling", "GlobalAvgPoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input",
                                             {batch, channels, height, width});

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
}

#define BM_GLOBAL_AVG_POOLING_MACRO(N, C, H, W, DEVICE)               \
  static void BM_GLOBAL_AVG_POOLING_##N##_##C##_##H##_##W##_##DEVICE( \
      int iters) {                                                    \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;  \
    mace::testing::ItemsProcessed(tot);                               \
    mace::testing::BytesProcessed(tot *(sizeof(float)));              \
    GlobalAvgPooling<DEVICE>(iters, N, C, H, W);                      \
  }                                                                   \
  BENCHMARK(BM_GLOBAL_AVG_POOLING_##N##_##C##_##H##_##W##_##DEVICE)

#define BM_GLOBAL_AVG_POOLING(N, C, H, W)       \
  BM_GLOBAL_AVG_POOLING_MACRO(N, C, H, W, CPU); \
  BM_GLOBAL_AVG_POOLING_MACRO(N, C, H, W, NEON);

BM_GLOBAL_AVG_POOLING(1, 3, 7, 7);
BM_GLOBAL_AVG_POOLING(1, 3, 64, 64);
BM_GLOBAL_AVG_POOLING(1, 3, 256, 256);