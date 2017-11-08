//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void BMBatchToSpace(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("BatchToSpaceND", "BatchToSpaceNDTest")
      .Input("Input")
      .Input("BlockShape")
      .Input("Crops")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
  net.AddInputFromArray<D, int>(
      "BlockShape", {2}, {2, 2});
  net.AddInputFromArray<D, int>("Crops", {2, 2}, {0,1,0,1});

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
  net.Sync();
}

#define BM_BATCH_TO_SPACE_MACRO(N, C, H, W, TYPE, DEVICE)                  \
  static void BM_BATCH_TO_SPACE_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE( \
      int iters) {                                                     \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;   \
    mace::testing::ItemsProcessed(tot);                                \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                \
    BMBatchToSpace<DEVICE, TYPE>(iters, N, C, H, W);                   \
  }                                                                    \
  BENCHMARK(BM_BATCH_TO_SPACE_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define BM_BATCH_TO_SPACE(N, C, H, W, TYPE)       \
  BM_BATCH_TO_SPACE_MACRO(N, C, H, W, TYPE, OPENCL);

BM_BATCH_TO_SPACE(128, 128, 8, 8, float);
}  //  namespace mace