//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void BMBatchToSpace(
    int iters, int batch, int channels, int height, int width, int arg) {
  mace::testing::StopTiming();

  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  BufferToImage<D, float>(net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("BatchToSpaceND", "BatchToSpaceNDTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntsArg("crops", {0, 0, 0, 0})
      .AddIntsArg("block_shape", {arg, arg})
      .Finalize(net.NewOperatorDef());

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

#define BM_BATCH_TO_SPACE_MACRO(N, H, W, C, ARG, TYPE, DEVICE)             \
  static void                                                              \
      BM_BATCH_TO_SPACE_##N##_##H##_##W##_##C##_##ARG##_##TYPE##_##DEVICE( \
          int iters) {                                                     \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;       \
    mace::testing::MaccProcessed(tot);                                     \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                    \
    BMBatchToSpace<DEVICE, TYPE>(iters, N, C, H, W, ARG);                  \
  }                                                                        \
  BENCHMARK(BM_BATCH_TO_SPACE_##N##_##H##_##W##_##C##_##ARG##_##TYPE##_##DEVICE)

#define BM_BATCH_TO_SPACE(N, H, W, C, ARG) \
  BM_BATCH_TO_SPACE_MACRO(N, H, W, C, ARG, float, OPENCL);

BM_BATCH_TO_SPACE(128, 8, 8, 128, 2);
BM_BATCH_TO_SPACE(4, 128, 128, 32, 2);
BM_BATCH_TO_SPACE(16, 64, 64, 32, 4);
}  // namespace mace
