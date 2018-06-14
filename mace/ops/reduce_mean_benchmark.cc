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
void ReduceMean(int iters, int batch, int channels,
                int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, T>("Input", {batch, height, width, channels});

  if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("ReduceMean", "ReduceMeanBM")
        .Input("InputImage")
        .AddIntsArg("axis", {1, 2})
        .Output("OutputImage")
        .Finalize(net.NewOperatorDef());
  } else {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("ReduceMean", "ReduceMeanBM")
        .Input("InputNCHW")
        .AddIntsArg("axis", {2, 3})
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
}  // namespace

#define MACE_BM_REDUCE_MEAN_MACRO(N, C, H, W, TYPE, DEVICE)       \
  static void                                                \
    MACE_BM_REDUCE_MEAN_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(\
      int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W; \
    mace::testing::MaccProcessed(tot);                               \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    ReduceMean<DEVICE, TYPE>(iters, N, C, H, W);        \
  }                                                                  \
  MACE_BENCHMARK(                                                         \
    MACE_BM_REDUCE_MEAN_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_REDUCE_MEAN(N, C, H, W)                 \
  MACE_BM_REDUCE_MEAN_MACRO(N, C, H, W, float, GPU);  \
  MACE_BM_REDUCE_MEAN_MACRO(N, C, H, W, half, GPU);   \
  MACE_BM_REDUCE_MEAN_MACRO(N, C, H, W, float, CPU);


MACE_BM_REDUCE_MEAN(1, 1, 512, 512);
MACE_BM_REDUCE_MEAN(4, 3, 128, 128);
MACE_BM_REDUCE_MEAN(4, 3, 512, 512);
MACE_BM_REDUCE_MEAN(16, 32, 112, 112);
MACE_BM_REDUCE_MEAN(8, 32, 112, 112);
MACE_BM_REDUCE_MEAN(8, 64, 256, 256);
MACE_BM_REDUCE_MEAN(1, 32, 480, 640);


}  // namespace test
}  // namespace ops
}  // namespace mace
