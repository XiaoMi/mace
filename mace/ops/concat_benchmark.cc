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
      .AddIntArg("axis", concat_dim)
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  const int kDim0 = 100;
  net.AddRandomInput<DeviceType::CPU, T>("Input0", {kDim0, dim1});
  net.AddRandomInput<DeviceType::CPU, T>("Input1", {kDim0, dim1});

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

static void BM_CONCAT_Dim0Float(int iters, int dim1) {
  ConcatHelper<DeviceType::CPU, float>(iters, 0, dim1);
}

static void BM_CONCAT_Dim1Float(int iters, int dim1) {
  ConcatHelper<DeviceType::CPU, float>(iters, 1, dim1);
}
BENCHMARK(BM_CONCAT_Dim0Float)->Arg(1000)->Arg(100000);
BENCHMARK(BM_CONCAT_Dim1Float)->Arg(1000)->Arg(100000);

template <typename T>
static void OpenclConcatHelper(int iters,
                               const std::vector<index_t> &shape0,
                               const std::vector<index_t> &shape1,
                               int concat_dim) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>("Input0", shape0);
  net.AddRandomInput<DeviceType::OPENCL, float>("Input1", shape1);

  BufferToImage<DeviceType::OPENCL, T>(net, "Input0", "InputImage0",
                                       kernels::BufferType::IN_OUT);
  BufferToImage<DeviceType::OPENCL, T>(net, "Input1", "InputImage1",
                                       kernels::BufferType::IN_OUT);
  OpDefBuilder("Concat", "ConcatBM")
      .Input("InputImage0")
      .Input("InputImage1")
      .AddIntArg("axis", concat_dim)
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(DeviceType::OPENCL);
  }

  const int64_t tot =
      static_cast<int64_t>(iters) *
      (net.GetTensor("Input0")->size() + net.GetTensor("Input1")->size());
  mace::testing::ItemsProcessed(tot);
  testing::BytesProcessed(tot * sizeof(T));
  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(DeviceType::OPENCL);
  }
}

static void BM_CONCATOPENCLFloat(int iters, int dim1) {
  std::vector<index_t> shape = {3, 32, 32, dim1};
  OpenclConcatHelper<float>(iters, shape, shape, 3);
}

static void BM_CONCATOPENCLHalf(int iters, int dim1) {
  std::vector<index_t> shape = {3, 32, 32, dim1};
  OpenclConcatHelper<half>(iters, shape, shape, 3);
}

BENCHMARK(BM_CONCATOPENCLFloat)->Arg(32)->Arg(64)->Arg(128)->Arg(256);
BENCHMARK(BM_CONCATOPENCLHalf)->Arg(32)->Arg(64)->Arg(128)->Arg(256);

}  //  namespace mace