//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void ReluBenchmark(int iters, int size) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("Relu", "ReluBM")
      .Input("Input")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, float>("Input", {size});

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

#define BM_RELU_MACRO(SIZE, TYPE, DEVICE)                     \
  static void BM_RELU_##SIZE##_##TYPE##_##DEVICE(int iters) { \
    const int64_t tot = static_cast<int64_t>(iters) * SIZE;   \
    mace::testing::ItemsProcessed(tot);                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));       \
    ReluBenchmark<DEVICE, TYPE>(iters, SIZE);                 \
  }                                                           \
  BENCHMARK(BM_RELU_##SIZE##_##TYPE##_##DEVICE)

#define BM_RELU(SIZE, TYPE)       \
  BM_RELU_MACRO(SIZE, TYPE, CPU); \
  BM_RELU_MACRO(SIZE, TYPE, NEON);\
  BM_RELU_MACRO(SIZE, TYPE, OPENCL);

BM_RELU(1000, float);
BM_RELU(100000, float);
BM_RELU(10000000, float);
}  //  namespace mace