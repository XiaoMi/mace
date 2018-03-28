//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

template <DeviceType D, typename T>
static void DepthToSpace(
    int iters, int batch, int channels, int height, int width, int block_size) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("DepthToSpace", "DepthToSpaceBM")
        .Input("InputImage")
        .Output("Output")
        .AddIntArg("block_size", block_size)
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("DepthToSpace", "DepthToSpaceBM")
        .Input("Input")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
  }

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

#define BM_DEPTH_TO_SPACE_MACRO(N, C, H, W, G, TYPE, DEVICE)             \
  static void                                                            \
      BM_DEPTH_TO_SPACE_##N##_##C##_##H##_##W##_##G##_##TYPE##_##DEVICE( \
          int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;     \
    mace::testing::MaccProcessed(tot);                                   \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                  \
    DepthToSpace<DEVICE, TYPE>(iters, N, C, H, W, G);                    \
  }                                                                      \
  BENCHMARK(BM_DEPTH_TO_SPACE_##N##_##C##_##H##_##W##_##G##_##TYPE##_##DEVICE)

#define BM_DEPTH_TO_SPACE(N, C, H, W, G)                 \
  BM_DEPTH_TO_SPACE_MACRO(N, C, H, W, G, float, CPU);    \
  BM_DEPTH_TO_SPACE_MACRO(N, C, H, W, G, float, OPENCL); \
  BM_DEPTH_TO_SPACE_MACRO(N, C, H, W, G, half, OPENCL);

BM_DEPTH_TO_SPACE(1, 64, 64, 64, 4);
BM_DEPTH_TO_SPACE(1, 64, 128, 128, 4);
BM_DEPTH_TO_SPACE(1, 64, 256, 256, 4);

}  // namespace test
}  // namespace ops
}  // namespace mace
