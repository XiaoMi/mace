//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void GEMMBenchmark(
    int iters, int batch, int height, int channels, int out_width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("A", {batch, height, 1, channels});
  net.AddRandomInput<D, float>("B", {batch, channels, 1, out_width});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "A", "AImage",
                            kernels::BufferType::IN_OUT);
    BufferToImage<D, T>(net, "B", "BImage",
                            kernels::BufferType::IN_OUT);

    OpDefBuilder("GEMM", "GEMMBM")
        .Input("AImage")
        .Input("BImage")
        .Output("Output")
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("GEMM", "GEMMBM")
        .Input("A")
        .Input("B")
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

#define BM_GEMM_MACRO(N, H, C, W, TYPE, DEVICE)                      \
  static void BM_GEMM_##N##H##C##W##_##TYPE##_##DEVICE(int iters) {  \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W; \
    mace::testing::ItemsProcessed(tot);                              \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    GEMMBenchmark<DEVICE, TYPE>(iters, N, H, C, W);                  \
  }                                                                  \
  BENCHMARK(BM_GEMM_##N##H##C##W##_##TYPE##_##DEVICE)

#define BM_GEMM(N, H, C, W, TYPE)        \
  BM_GEMM_MACRO(N, H, C, W, TYPE, OPENCL);

BM_GEMM(16, 32, 128, 1024, half);
BM_GEMM(36, 32, 128, 256, half);
}  //  namespace mace
