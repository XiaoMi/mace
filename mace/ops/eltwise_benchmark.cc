//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/eltwise.h"
#include <string>
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void EltwiseBenchmark(
    int iters, kernels::EltwiseType type, int n, int h, int w, int c) {
  mace::testing::StopTiming();

  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, T>("Input0", {n, h, w, c});
  net.AddRandomInput<D, T>("Input1", {n, h, w, c});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, half>(net, "Input0", "InputImg0",
                           kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, half>(net, "Input1", "InputImg1",
                           kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("InputImg0")
        .Input("InputImg1")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatsArg("coeff", {1.2, 2.1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Output("OutputImg")
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Eltwise", "EltwiseTest")
        .Input("Input0")
        .Input("Input1")
        .AddIntArg("type", static_cast<int>(type))
        .AddFloatsArg("coeff", {1.2, 2.1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Output("Output")
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

#define BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, TYPE, DEVICE)             \
  static void                                                            \
      BM_ELTWISE_##ELT_TYPE##_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE( \
          int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C;     \
    mace::testing::MaccProcessed(tot);                                  \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                  \
    EltwiseBenchmark<DEVICE, TYPE>(                                      \
        iters, static_cast<kernels::EltwiseType>(ELT_TYPE), N, H, W, C); \
  }                                                                      \
  BENCHMARK(BM_ELTWISE_##ELT_TYPE##_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE)

#define BM_ELTWISE(ELT_TYPE, N, H, W, C)                 \
  BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, float, CPU);    \
  BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, float, OPENCL); \
  BM_ELTWISE_MACRO(ELT_TYPE, N, H, W, C, half, OPENCL);

BM_ELTWISE(0, 1, 256, 256, 32);
BM_ELTWISE(0, 1, 128, 128, 32);
BM_ELTWISE(1, 1, 128, 128, 32);
BM_ELTWISE(2, 1, 128, 128, 32);
BM_ELTWISE(0, 1, 240, 240, 256);
BM_ELTWISE(1, 1, 240, 240, 256);
BM_ELTWISE(2, 1, 240, 240, 256);

}  // namespace mace
