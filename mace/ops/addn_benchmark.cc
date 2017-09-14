//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void AddNBenchmark(int iters, int n, int size) {

  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder op_def_builder("AddN", "AddNBM");
  for (int i = 0; i < n; ++i) {
    op_def_builder.Input(internal::MakeString("Input", i).c_str());
  }
  op_def_builder.Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  for (int i = 0; i < n; ++i) {
    net.AddRandomInput<float>(internal::MakeString("Input", i).c_str(), {size});
  }

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }

  mace::testing::StartTiming();
  while(iters--) {
    net.RunOp(D);
  }
}

#define BM_ADDN_MACRO(N, SIZE, TYPE, DEVICE)                       \
  static void BM_ADDN_##N##_##SIZE##_##TYPE##_##DEVICE(            \
        int iters) {                                               \
    const int64_t tot = static_cast<int64_t>(iters) * N * SIZE;    \
    mace::testing::ItemsProcessed(tot);                            \
    mace::testing::BytesProcessed(tot * (sizeof(TYPE)));           \
    AddNBenchmark<DEVICE, TYPE>(iters, N, SIZE);                   \
  }                                                                \
  BENCHMARK(BM_ADDN_##N##_##SIZE##_##TYPE##_##DEVICE)

#define BM_ADDN(N, SIZE, TYPE)        \
  BM_ADDN_MACRO(N, SIZE, TYPE, CPU);  \
  BM_ADDN_MACRO(N, SIZE, TYPE, NEON);

BM_ADDN(10, 1000, float);
BM_ADDN(10, 10000, float);
BM_ADDN(100, 1000, float);
BM_ADDN(100, 10000, float);
} //  namespace mace