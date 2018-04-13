//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void ChannelShuffle(
    int iters, int batch, int channels, int height, int width, int group) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("ChannelShuffle", "ChannelShuffleTest")
        .Input("InputImage")
        .Output("Output")
        .AddIntArg("group", group)
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Softmax", "SoftmaxBM")
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
}  // namespace

#define BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, TYPE, DEVICE)             \
  static void                                                             \
      BM_CHANNEL_SHUFFLE_##N##_##C##_##H##_##W##_##G##_##TYPE##_##DEVICE( \
          int iters) {                                                    \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;      \
    mace::testing::MaccProcessed(tot);                                    \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    ChannelShuffle<DEVICE, TYPE>(iters, N, C, H, W, G);                   \
  }                                                                       \
  BENCHMARK(BM_CHANNEL_SHUFFLE_##N##_##C##_##H##_##W##_##G##_##TYPE##_##DEVICE)

#define BM_CHANNEL_SHUFFLE(N, C, H, W, G)                 \
  BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, float, CPU);    \
  BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, float, OPENCL); \
  BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, half, OPENCL);

BM_CHANNEL_SHUFFLE(1, 64, 64, 64, 8);
BM_CHANNEL_SHUFFLE(1, 64, 128, 128, 8);
BM_CHANNEL_SHUFFLE(1, 64, 256, 256, 8);

}  // namespace test
}  // namespace ops
}  // namespace mace
