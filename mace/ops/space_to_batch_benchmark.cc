//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void BMSpaceToBatch(
    int iters, int batch, int height, int width, int channels, int shape) {
  mace::testing::StopTiming();

  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  BufferToImage<D, float>(net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntsArg("paddings", {shape, shape, shape, shape})
      .AddIntsArg("block_shape", {shape, shape})
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

#define BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, TYPE, DEVICE)             \
  static void                                                                \
      BM_SPACE_TO_BATCH_##N##_##H##_##W##_##C##_##SHAPE##_##TYPE##_##DEVICE( \
          int iters) {                                                       \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;         \
    mace::testing::MaccProcessed(tot);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                      \
    BMSpaceToBatch<DEVICE, TYPE>(iters, N, H, W, C, SHAPE);                  \
  }                                                                          \
  BENCHMARK(                                                                 \
      BM_SPACE_TO_BATCH_##N##_##H##_##W##_##C##_##SHAPE##_##TYPE##_##DEVICE)

#define BM_SPACE_TO_BATCH(N, H, W, C, SHAPE) \
  BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, float, OPENCL);

BM_SPACE_TO_BATCH(128, 16, 16, 128, 2);
BM_SPACE_TO_BATCH(1, 256, 256, 32, 2);
BM_SPACE_TO_BATCH(1, 256, 256, 32, 4);
}  // namespace mace
