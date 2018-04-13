//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void Pad(int iters, int batch, int height,
         int width, int channels, int pad) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, T>("Input", {batch, height, width, channels});

  const std::vector<int> paddings = {0, 0, pad, pad, pad, pad, 0, 0};
  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("Pad", "PadTest")
        .Input("InputImage")
        .Output("OutputImage")
        .AddIntsArg("paddings", paddings)
        .AddFloatArg("constant_value", 1.0)
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Pad", "PadTest")
        .Input("Input")
        .Output("Output")
        .AddIntsArg("paddings", paddings)
        .AddFloatArg("constant_value", 1.0)
        .Finalize(net.NewOperatorDef());
  }

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}
}  // namespace

#define BM_PAD_MACRO(N, H, W, C, PAD, TYPE, DEVICE)                  \
  static void BM_PAD_##N##_##H##_##W##_##C##_##PAD##_##TYPE##_##DEVICE( \
      int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W; \
    mace::testing::MaccProcessed(tot);                               \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    Pad<DEVICE, TYPE>(iters, N, H, W, C, PAD);                       \
  }                                                                  \
  BENCHMARK(BM_PAD_##N##_##H##_##W##_##C##_##PAD##_##TYPE##_##DEVICE)

#define BM_PAD(N, H, W, C, PAD)                 \
  BM_PAD_MACRO(N, H, W, C, PAD, float, CPU);    \
  BM_PAD_MACRO(N, H, W, C, PAD, float, OPENCL); \
  BM_PAD_MACRO(N, H, W, C, PAD, half, OPENCL);

BM_PAD(1, 512, 512, 1, 2);
BM_PAD(1, 112, 112, 64, 1);
BM_PAD(1, 256, 256, 32, 2);
BM_PAD(1, 512, 512, 16, 2);

}  // namespace test
}  // namespace ops
}  // namespace mace
