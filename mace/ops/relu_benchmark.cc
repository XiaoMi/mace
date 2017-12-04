//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void ReluBenchmark(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);

    OpDefBuilder("Relu", "ReluBM")
        .Input("InputImage")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Relu", "ReluBM")
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

#define BM_RELU_MACRO(N, C, H, W, TYPE, DEVICE)                      \
  static void BM_RELU_##N##C##H##W##_##TYPE##_##DEVICE(int iters) {  \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W; \
    mace::testing::ItemsProcessed(tot);                              \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    ReluBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                  \
  }                                                                  \
  BENCHMARK(BM_RELU_##N##C##H##W##_##TYPE##_##DEVICE)

#define BM_RELU(N, C, H, W, TYPE)       \
  BM_RELU_MACRO(N, C, H, W, TYPE, CPU); \
  BM_RELU_MACRO(N, C, H, W, TYPE, NEON);\
  BM_RELU_MACRO(N, C, H, W, TYPE, OPENCL);

BM_RELU(1, 1, 512, 512, float);
BM_RELU(1, 3, 128, 128, float);
BM_RELU(1, 3, 512, 512, float);
BM_RELU(1, 32, 112, 112, float);
BM_RELU(1, 64, 256, 256, float);
}  //  namespace mace
