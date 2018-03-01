//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void FCBenchmark(
    int iters, int batch, int height, int width, int channel, int out_channel) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channel});
  net.AddRandomInput<D, float>("Weight", {out_channel, height * width * channel});
  net.AddRandomInput<D, float>("Bias", {out_channel});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(net, "Weight", "WeightImage",
                            kernels::BufferType::WEIGHT_HEIGHT);
    BufferToImage<D, T>(net, "Bias", "BiasImage",
                            kernels::BufferType::ARGUMENT);

    OpDefBuilder("FC", "FullyConnectedTest")
        .Input("InputImage")
        .Input("WeightImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("FC", "FullyConnectedTest")
        .Input("Input")
        .Input("Weight")
        .Input("Bias")
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

#define BM_FC_MACRO(N, H, W, C, OC, TYPE, DEVICE)                              \
  static void BM_FC_##N##_##H##_##W##_##C##_##OC##_##TYPE##_##DEVICE(int iters) { \
    const int64_t macc = static_cast<int64_t>(iters) * N * C * H * W * OC + OC;  \
    const int64_t tot = static_cast<int64_t>(iters) * (N + OC) * C * H * W + OC; \
    mace::testing::MaccProcessed(macc);                                          \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                          \
    FCBenchmark<DEVICE, TYPE>(iters, N, H, W, C, OC);                            \
  }                                                                              \
  BENCHMARK(BM_FC_##N##_##H##_##W##_##C##_##OC##_##TYPE##_##DEVICE)

#define BM_FC(N, H, W, C, OC)                 \
  BM_FC_MACRO(N, H, W, C, OC, float, CPU);    \
  BM_FC_MACRO(N, H, W, C, OC, float, OPENCL); \
  BM_FC_MACRO(N, H, W, C, OC, half, OPENCL);

BM_FC(1, 16, 16, 32, 32);
BM_FC(1, 8, 8, 32, 1000);
}  // namespace mace
