//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void BMWinogradTransform(
    int iters, int batch, int height, int width, int channels) {
  mace::testing::StopTiming();

  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  BufferToImage<D, T>(net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("WinogradTransform", "WinogradTransformTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

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

#define BM_WINOGRAD_TRANSFORM_MACRO(N, H, W, C, TYPE, DEVICE)             \
  static void                                                         \
      BM_WINOGRAD_TRANSFORM_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE(    \
          int iters) {                                                \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;  \
    mace::testing::ItemsProcessed(tot);                               \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));               \
    BMWinogradTransform<DEVICE, TYPE>(iters, N, H, W, C);                  \
  }                                                                   \
  BENCHMARK(                                                          \
      BM_WINOGRAD_TRANSFORM_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE)

#define BM_WINOGRAD_TRANSFORM(N, H, W, C, TYPE) \
  BM_WINOGRAD_TRANSFORM_MACRO(N, H, W, C, TYPE, OPENCL);

BM_WINOGRAD_TRANSFORM(1, 16, 16, 128, half);
BM_WINOGRAD_TRANSFORM(1, 64, 64, 128, half);
BM_WINOGRAD_TRANSFORM(1, 128, 128, 128, half);

template <DeviceType D, typename T>
static void BMWinogradInverseTransform(
    int iters, int batch, int height, int width, int channels) {
  mace::testing::StopTiming();

  index_t p = batch * ((height + 1) / 2) * ((width + 1) / 2);
  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", {16, channels, p, 1});

  BufferToImage<D, T>(net, "Input", "InputImage",
                      kernels::BufferType::IN_OUT_HEIGHT);
  OpDefBuilder("WinogradInverseTransform", "WinogradInverseTransformTest")
      .Input("InputImage")
      .AddIntArg("batch", batch)
      .AddIntArg("height", height)
      .AddIntArg("width", width)
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

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

#define BM_WINOGRAD_INVERSE_TRANSFORM_MACRO(N, H, W, C, TYPE, DEVICE)             \
  static void                                                         \
      BM_WINOGRAD_INVERSE_TRANSFORM_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE(    \
          int iters) {                                                \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;  \
    mace::testing::ItemsProcessed(tot);                               \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));               \
    BMWinogradInverseTransform<DEVICE, TYPE>(iters, N, H, W, C);                  \
  }                                                                   \
  BENCHMARK(                                                          \
      BM_WINOGRAD_INVERSE_TRANSFORM_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE)

#define BM_WINOGRAD_INVERSE_TRANSFORM(N, H, W, C, TYPE) \
  BM_WINOGRAD_INVERSE_TRANSFORM_MACRO(N, H, W, C, TYPE, OPENCL);

BM_WINOGRAD_INVERSE_TRANSFORM(1, 14, 14, 32, half);
BM_WINOGRAD_INVERSE_TRANSFORM(1, 62, 62, 32, half);
BM_WINOGRAD_INVERSE_TRANSFORM(1, 126, 126, 32, half);

}  //  namespace mace