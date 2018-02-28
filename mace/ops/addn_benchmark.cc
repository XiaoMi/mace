//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void AddNBenchmark(int iters, int inputs, int n, int h, int w, int c) {
  mace::testing::StopTiming();

  OpsTestNet net;
  // Add input data
  for (int i = 0; i < inputs; ++i) {
    net.AddRandomInput<D, float>(MakeString("Input", i).c_str(),
                                 {n, h, w, c});
  }

  if (D == DeviceType::OPENCL) {
    for (int i = 0; i < inputs; ++i) {
      BufferToImage<D, T>(net, MakeString("Input", i).c_str(),
                          MakeString("InputImage", i).c_str(),
                          kernels::BufferType::IN_OUT_CHANNEL);
    }
    OpDefBuilder op_def_builder("AddN", "AddNBM");
    for (int i = 0; i < inputs; ++i) {
      op_def_builder.Input(MakeString("InputImage", i).c_str());
    }
    op_def_builder.Output("OutputImage")
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder op_def_builder("AddN", "AddNBM");
    for (int i = 0; i < inputs; ++i) {
      op_def_builder.Input(MakeString("Input", i).c_str());
    }
    op_def_builder.Output("Output")
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  }

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
    net.Sync();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
    net.Sync();
  }
}

#define BM_ADDN_MACRO(INPUTS, N, H, W, C, TYPE, DEVICE)                       \
  static void BM_ADDN_##INPUTS##_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE(   \
      int iters) {                                                            \
    const int64_t tot = static_cast<int64_t>(iters) * INPUTS * N * H * W * C; \
    mace::testing::MaccProcessed(tot);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    AddNBenchmark<DEVICE, TYPE>(iters, INPUTS, N, H, W, C);                   \
  }                                                                           \
  BENCHMARK(BM_ADDN_##INPUTS##_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE)

#define BM_ADDN(INPUTS, N, H, W, C)                 \
  BM_ADDN_MACRO(INPUTS, N, H, W, C, float, CPU);    \
  BM_ADDN_MACRO(INPUTS, N, H, W, C, float, OPENCL); \
  BM_ADDN_MACRO(INPUTS, N, H, W, C, half, OPENCL);

BM_ADDN(2, 1, 256, 256, 32);
BM_ADDN(2, 1, 128, 128, 32);
BM_ADDN(4, 1, 128, 128, 3);
BM_ADDN(2, 1, 256, 256, 3);
BM_ADDN(2, 1, 512, 512, 3);

}  // namespace mace
