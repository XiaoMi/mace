//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void BatchNorm(int iters, int batch, int channels, int height, int width) {

  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("BatchNorm", "BatchNormBM")
          .Input("Input")
          .Input("Scale")
          .Input("Offset")
          .Input("Mean")
          .Input("Var")
          .Output("Output")
          .Finalize(net.operator_def());

  // Add input data
  net.AddRandomInput<T>("Input", {batch, channels, height, width});
  net.AddRandomInput<T>("Scale", {channels});
  net.AddRandomInput<T>("Offset", {channels});
  net.AddRandomInput<T>("Mean", {channels});
  net.AddRandomInput<T>("Var", {channels}, true);

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }

  mace::testing::StartTiming();
  while(iters--) {
    net.RunOp(D);
  }
}

#define BM_BATCH_NORM_MACRO(N, C, H, W, TYPE, DEVICE)                   \
  static void BM_BATCH_NORM_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(  \
        int iters) {                                                    \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;    \
    mace::testing::ItemsProcessed(tot);                                 \
    mace::testing::BytesProcessed(tot * (sizeof(TYPE)));                \
    BatchNorm<DEVICE, TYPE>(iters, N, C, H, W);                         \
  }                                                                     \
  BENCHMARK(BM_BATCH_NORM_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define BM_BATCH_NORM(N, C, H, W, TYPE)        \
  BM_BATCH_NORM_MACRO(N, C, H, W, TYPE, CPU);  \
  BM_BATCH_NORM_MACRO(N, C, H, W, TYPE, NEON);

BM_BATCH_NORM(1, 1, 512, 512, float);
BM_BATCH_NORM(1, 1, 1024, 1024, float);
BM_BATCH_NORM(1, 3, 128, 128, float);
BM_BATCH_NORM(1, 3, 512, 512, float);
BM_BATCH_NORM(1, 3, 1024, 1024, float);
BM_BATCH_NORM(1, 64, 256, 256, float);
BM_BATCH_NORM(1, 64, 512, 512, float);
BM_BATCH_NORM(1, 128, 256, 256, float);
BM_BATCH_NORM(1, 128, 512, 512, float);
BM_BATCH_NORM(32, 1, 256, 256, float);
BM_BATCH_NORM(32, 3, 256, 256, float);
} //  namespace mace