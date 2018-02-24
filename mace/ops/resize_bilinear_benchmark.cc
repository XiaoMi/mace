//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void ResizeBilinearBenchmark(int iters,
                                    int batch,
                                    int channels,
                                    int input_height,
                                    int input_width,
                                    int output_height,
                                    int output_width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input",
                               {batch, input_height, input_width, channels});
  net.AddInputFromArray<D, index_t>("OutSize", {2},
                                    {output_height, output_width});
  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder("ResizeBilinear", "ResizeBilinearBenchmark")
        .Input("InputImage")
        .Input("OutSize")
        .Output("OutputImage")
        .AddIntsArg("size", {output_height, output_width})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("ResizeBilinear", "ResizeBilinearBenchmark")
        .Input("Input")
        .Input("OutSize")
        .Output("Output")
        .AddIntsArg("size", {output_height, output_width})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  }

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
  net.Sync();
}

#define BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, TYPE, DEVICE)                \
  static void                                                                       \
      BM_RESIZE_BILINEAR_##N##_##C##_##H0##_##W0##_##H1##_##W1##_##TYPE##_##DEVICE( \
          int iters) {                                                              \
    const int64_t macc = static_cast<int64_t>(iters) * N * C * H1 * W1 * 3;         \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H0 * W0;              \
    mace::testing::MaccProcessed(macc);                                             \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                             \
    ResizeBilinearBenchmark<DEVICE, TYPE>(iters, N, C, H0, W0, H1, W1);             \
  }                                                                                 \
  BENCHMARK(                                                                        \
      BM_RESIZE_BILINEAR_##N##_##C##_##H0##_##W0##_##H1##_##W1##_##TYPE##_##DEVICE)

#define BM_RESIZE_BILINEAR(N, C, H0, W0, H1, W1)                 \
  BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, float, CPU);    \
  BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, float, OPENCL); \
  BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, half, OPENCL);

BM_RESIZE_BILINEAR(1, 128, 120, 120, 480, 480);

BM_RESIZE_BILINEAR(1, 256, 7, 7, 15, 15);
BM_RESIZE_BILINEAR(1, 256, 15, 15, 30, 30);
BM_RESIZE_BILINEAR(1, 128, 30, 30, 60, 60);
BM_RESIZE_BILINEAR(1, 128, 240, 240, 480, 480);
BM_RESIZE_BILINEAR(1, 3, 4032, 3016, 480, 480);
BM_RESIZE_BILINEAR(1, 3, 480, 480, 4032, 3016);

}  // namespace mace
