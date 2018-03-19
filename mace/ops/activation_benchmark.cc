//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <string>
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
template <DeviceType D, typename T>
static void ReluBenchmark(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "ReluBM")
        .Input("InputImage")
        .Output("Output")
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Activation", "ReluBM")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "RELU")
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

#define BM_RELU_MACRO(N, C, H, W, TYPE, DEVICE)                              \
  static void BM_RELU_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(int iters) { \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;         \
    mace::testing::MaccProcessed(tot);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                      \
    ReluBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                          \
  }                                                                          \
  BENCHMARK(BM_RELU_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define BM_RELU(N, C, H, W)                 \
  BM_RELU_MACRO(N, C, H, W, float, CPU);    \
  BM_RELU_MACRO(N, C, H, W, float, OPENCL); \
  BM_RELU_MACRO(N, C, H, W, half, OPENCL);

BM_RELU(1, 1, 512, 512);
BM_RELU(1, 3, 128, 128);
BM_RELU(1, 3, 512, 512);
BM_RELU(1, 32, 112, 112);
BM_RELU(1, 64, 256, 256);

template <DeviceType D, typename T>
static void ReluxBenchmark(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "ReluxBM")
        .Input("InputImage")
        .Output("Output")
        .AddStringArg("activation", "RELUX")
        .AddFloatArg("max_limit", 6.0)
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Activation", "ReluxBM")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "RELUX")
        .AddFloatArg("max_limit", 6.0)
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

#define BM_RELUX_MACRO(N, C, H, W, TYPE, DEVICE)                              \
  static void BM_RELUX_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(int iters) { \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;          \
    mace::testing::MaccProcessed(tot);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    ReluxBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                          \
  }                                                                           \
  BENCHMARK(BM_RELUX_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define BM_RELUX(N, C, H, W)                 \
  BM_RELUX_MACRO(N, C, H, W, float, CPU);    \
  BM_RELUX_MACRO(N, C, H, W, float, OPENCL); \
  BM_RELUX_MACRO(N, C, H, W, half, OPENCL);

BM_RELUX(1, 1, 512, 512);
BM_RELUX(1, 3, 128, 128);
BM_RELUX(1, 3, 512, 512);
BM_RELUX(1, 32, 112, 112);
BM_RELUX(1, 64, 256, 256);

template <DeviceType D, typename T>
static void PreluBenchmark(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  net.AddRandomInput<D, float>("Alpha", {channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, float>(&net, "Alpha", "AlphaImage",
                            kernels::BufferType::ARGUMENT);

    OpDefBuilder("Activation", "PreluBM")
        .Input("InputImage")
        .Input("AlphaImage")
        .Output("Output")
        .AddStringArg("activation", "PRELU")
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Activation", "PreluBM")
        .Input("Input")
        .Input("Alpha")
        .Output("Output")
        .AddStringArg("activation", "PRELU")
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

#define BM_PRELU_MACRO(N, C, H, W, TYPE, DEVICE)                              \
  static void BM_PRELU_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(int iters) { \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;          \
    mace::testing::MaccProcessed(tot);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    PreluBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                          \
  }                                                                           \
  BENCHMARK(BM_PRELU_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define BM_PRELU(N, C, H, W)                 \
  BM_PRELU_MACRO(N, C, H, W, float, CPU);    \
  BM_PRELU_MACRO(N, C, H, W, float, OPENCL); \
  BM_PRELU_MACRO(N, C, H, W, half, OPENCL);

BM_PRELU(1, 1, 512, 512);
BM_PRELU(1, 3, 128, 128);
BM_PRELU(1, 3, 512, 512);
BM_PRELU(1, 32, 112, 112);
BM_PRELU(1, 64, 256, 256);

template <DeviceType D, typename T>
static void TanhBenchmark(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "TanhBM")
        .Input("InputImage")
        .Output("Output")
        .AddStringArg("activation", "TANH")
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Activation", "TanhBM")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "TANH")
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

#define BM_TANH_MACRO(N, C, H, W, TYPE, DEVICE)                              \
  static void BM_TANH_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(int iters) { \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;         \
    mace::testing::MaccProcessed(tot);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                      \
    TanhBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                          \
  }                                                                          \
  BENCHMARK(BM_TANH_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define BM_TANH(N, C, H, W)                 \
  BM_TANH_MACRO(N, C, H, W, float, CPU);    \
  BM_TANH_MACRO(N, C, H, W, float, OPENCL); \
  BM_TANH_MACRO(N, C, H, W, half, OPENCL);

BM_TANH(1, 1, 512, 512);
BM_TANH(1, 3, 128, 128);
BM_TANH(1, 3, 512, 512);
BM_TANH(1, 32, 112, 112);
BM_TANH(1, 64, 256, 256);

template <DeviceType D, typename T>
static void SigmoidBenchmark(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "SigmoidBM")
        .Input("InputImage")
        .Output("Output")
        .AddStringArg("activation", "SIGMOID")
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Activation", "SigmoidBM")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "SIGMOID")
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

#define BM_SIGMOID_MACRO(N, C, H, W, TYPE, DEVICE)                   \
  static void BM_SIGMOID_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(  \
      int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W; \
    mace::testing::MaccProcessed(tot);                               \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    SigmoidBenchmark<DEVICE, TYPE>(iters, N, C, H, W);               \
  }                                                                  \
  BENCHMARK(BM_SIGMOID_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define BM_SIGMOID(N, C, H, W)                 \
  BM_SIGMOID_MACRO(N, C, H, W, float, CPU);    \
  BM_SIGMOID_MACRO(N, C, H, W, float, OPENCL); \
  BM_SIGMOID_MACRO(N, C, H, W, half, OPENCL);

BM_SIGMOID(1, 1, 512, 512);
BM_SIGMOID(1, 3, 128, 128);
BM_SIGMOID(1, 3, 512, 512);
BM_SIGMOID(1, 32, 112, 112);
BM_SIGMOID(1, 64, 256, 256);

}  // namespace mace
