//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void SoftmaxBenchmark(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Softmax", "SoftmaxBM")
        .Input("InputImage")
        .Output("Output")
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

#define BM_SOFTMAX_MACRO(N, C, H, W, TYPE, DEVICE)                      \
  static void BM_SOFTMAX_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(int iters) {  \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W; \
    mace::testing::MaccProcessed(tot);                              \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    SoftmaxBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                  \
  }                                                                  \
  BENCHMARK(BM_SOFTMAX_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define BM_SOFTMAX(N, C, H, W)                 \
  BM_SOFTMAX_MACRO(N, C, H, W, float, CPU);    \
  BM_SOFTMAX_MACRO(N, C, H, W, float, OPENCL); \
  BM_SOFTMAX_MACRO(N, C, H, W, half, OPENCL);

BM_SOFTMAX(1, 1, 512, 512);
BM_SOFTMAX(1, 3, 128, 128);
BM_SOFTMAX(1, 3, 512, 512);
BM_SOFTMAX(1, 32, 112, 112);
BM_SOFTMAX(1, 64, 256, 256);
}  // namespace mace
