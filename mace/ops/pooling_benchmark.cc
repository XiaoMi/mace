//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/pooling.h"
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

template <DeviceType D>
static void Pooling(int iters,
                    int batch,
                    int channels,
                    int height,
                    int width,
                    int kernel,
                    int stride,
                    Padding padding,
                    PoolingType pooling_type) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", pooling_type)
      .AddIntsArg("kernels", {kernel, kernel})
      .AddIntsArg("strides", {stride, stride})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input",
                                             {batch, channels, height, width});

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
}

#define BM_POOLING_MACRO(N, C, H, W, KE, STRIDE, PA, PO, DEVICE)          \
  static void                                                             \
      BM_POOLING_##N##_##C##_##H##_##W##_K##KE##S##STRIDE##_##PA##_##PO##_\
        ##DEVICE(                                                         \
          int iters) {                                                    \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;      \
    mace::testing::MaccProcessed(tot);                                    \
    mace::testing::BytesProcessed(tot *(sizeof(float)));                  \
    Pooling<DEVICE>(iters, N, C, H, W, KE, STRIDE, Padding::PA,           \
                    PoolingType::PO);                                     \
  }                                                                       \
  BENCHMARK(                                                              \
      BM_POOLING_##N##_##C##_##H##_##W##_K##KE##S##STRIDE##_##PA##_##PO##_\
        ##DEVICE)

#define BM_POOLING(N, C, H, W, K, S, PA, PO) \
  BM_POOLING_MACRO(N, C, H, W, K, S, PA, PO, CPU);
//  BM_POOLING_MACRO(N, C, H, W, K, S, PA, PO, NEON);

BM_POOLING(1, 3, 129, 129, 2, 2, SAME, MAX);
BM_POOLING(1, 3, 257, 257, 2, 2, SAME, MAX);
BM_POOLING(1, 3, 513, 513, 2, 2, SAME, MAX);
BM_POOLING(1, 3, 1025, 1025, 2, 2, SAME, MAX);

}  // namespace test
}  // namespace ops
}  // namespace mace
