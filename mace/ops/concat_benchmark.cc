//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void ConcatHelper(int iters, int concat_dim, int dim1) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("Concat", "ConcatBM")
      .Input("Input0")
      .Input("Input1")
      .Input("Axis")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  const int kDim0 = 100;
  net.AddRandomInput<DeviceType::CPU, T>("Input0", {kDim0, dim1});
  net.AddRandomInput<DeviceType::CPU, T>("Input1", {kDim0, dim1});
  net.AddInputFromArray<DeviceType::CPU, int32_t>("Axis", {}, {concat_dim});

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }
  const int64_t tot = static_cast<int64_t>(iters) * kDim0 * dim1 * 2;
  mace::testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(T));
  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
}

static void BM_ConcatDim0Float(int iters, int dim1) {
  ConcatHelper<DeviceType::CPU, float>(iters, 0, dim1);
}

static void BM_ConcatDim1Float(int iters, int dim1) {
  ConcatHelper<DeviceType::CPU, float>(iters, 1, dim1);
}
BENCHMARK(BM_ConcatDim0Float)->Arg(1000)->Arg(100000);
BENCHMARK(BM_ConcatDim1Float)->Arg(1000)->Arg(100000);

}  //  namespace mace