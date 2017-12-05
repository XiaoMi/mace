//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void BiasAdd(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, T>("Input", {batch, height, width, channels});
  net.AddRandomInput<D, T>("Bias", {channels}, true);

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);
    BufferToImage<D, T>(net, "Bias", "BiasImage", kernels::BufferType::ARGUMENT);
    OpDefBuilder("BiasAdd", "BiasAddBM")
        .Input("InputImage")
        .Input("BiasImage")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
  }
  else {
    OpDefBuilder("BiasAdd", "BiasAddBM")
        .Input("Input")
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

#define BM_BIAS_ADD_MACRO(N, C, H, W, TYPE, DEVICE)                  \
  static void BM_BIAS_ADD_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE( \
      int iters) {                                                     \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;   \
    mace::testing::ItemsProcessed(tot);                                \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                \
    BiasAdd<DEVICE, TYPE>(iters, N, C, H, W);                        \
  }                                                                    \
  BENCHMARK(BM_BIAS_ADD_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define BM_BIAS_ADD(N, C, H, W, TYPE)       \
  BM_BIAS_ADD_MACRO(N, C, H, W, TYPE, CPU); \
  BM_BIAS_ADD_MACRO(N, C, H, W, TYPE, OPENCL);

BM_BIAS_ADD(1, 1, 512, 512, float);
BM_BIAS_ADD(1, 3, 128, 128, float);
BM_BIAS_ADD(1, 3, 512, 512, float);
BM_BIAS_ADD(1, 32, 112, 112, float);
BM_BIAS_ADD(1, 64, 256, 256, float);
BM_BIAS_ADD(1, 64, 512, 512, float);
BM_BIAS_ADD(1, 128, 56, 56, float);
BM_BIAS_ADD(1, 128, 256, 256, float);
BM_BIAS_ADD(1, 256, 14, 14, float);
BM_BIAS_ADD(1, 512, 14, 14, float);
BM_BIAS_ADD(1, 1024, 7, 7, float);
BM_BIAS_ADD(32, 1, 256, 256, float);
BM_BIAS_ADD(32, 3, 256, 256, float);
}  //  namespace mace
