// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void ReluBenchmark(int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
  } else if (D == DeviceType::GPU) {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  if (D == DeviceType::CPU) {
    OpDefBuilder("Activation", "ReluBM")
        .Input("Input")
        .Output("Output")
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);

    OpDefBuilder("Activation", "ReluBM")
        .Input("InputImage")
        .Output("Output")
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());
  } else {
    MACE_NOT_IMPLEMENTED;
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

#define MACE_BM_RELU_MACRO(N, C, H, W, TYPE, DEVICE)                         \
  static void MACE_BM_RELU_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(        \
      int iters) {                                                           \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;         \
    mace::testing::MaccProcessed(tot);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                      \
    ReluBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                          \
  }                                                                          \
  MACE_BENCHMARK(MACE_BM_RELU_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_RELU(N, C, H, W)              \
  MACE_BM_RELU_MACRO(N, C, H, W, float, CPU); \
  MACE_BM_RELU_MACRO(N, C, H, W, float, GPU); \
  MACE_BM_RELU_MACRO(N, C, H, W, half, GPU);

MACE_BM_RELU(1, 1, 512, 512);
MACE_BM_RELU(1, 3, 128, 128);
MACE_BM_RELU(1, 3, 512, 512);
MACE_BM_RELU(1, 32, 112, 112);
MACE_BM_RELU(1, 64, 256, 256);

namespace {
template <DeviceType D, typename T>
void ReluxBenchmark(int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
  } else {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  }

  if (D == DeviceType::GPU) {
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
}  // namespace

#define MACE_BM_RELUX_MACRO(N, C, H, W, TYPE, DEVICE)                         \
  static void MACE_BM_RELUX_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(        \
      int iters) {                                                            \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;          \
    mace::testing::MaccProcessed(tot);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    ReluxBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                          \
  }                                                                           \
  MACE_BENCHMARK(MACE_BM_RELUX_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_RELUX(N, C, H, W)              \
  MACE_BM_RELUX_MACRO(N, C, H, W, float, CPU); \
  MACE_BM_RELUX_MACRO(N, C, H, W, float, GPU); \
  MACE_BM_RELUX_MACRO(N, C, H, W, half, GPU);

MACE_BM_RELUX(1, 1, 512, 512);
MACE_BM_RELUX(1, 3, 128, 128);
MACE_BM_RELUX(1, 3, 512, 512);
MACE_BM_RELUX(1, 32, 112, 112);
MACE_BM_RELUX(1, 64, 256, 256);

namespace {
template <DeviceType D, typename T>
void PreluBenchmark(int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
  } else if (D == DeviceType::GPU) {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  net.AddRandomInput<D, float>("Alpha", {channels});

  if (D == DeviceType::CPU) {
    OpDefBuilder("Activation", "PreluBM")
        .Input("Input")
        .Input("Alpha")
        .Output("Output")
        .AddStringArg("activation", "PRELU")
        .Finalize(net.NewOperatorDef());
  } else if (D == DeviceType::GPU) {
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
    MACE_NOT_IMPLEMENTED;
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

#define MACE_BM_PRELU_MACRO(N, C, H, W, TYPE, DEVICE)                         \
  static void MACE_BM_PRELU_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(        \
      int iters) {                                                            \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;          \
    mace::testing::MaccProcessed(tot);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    PreluBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                          \
  }                                                                           \
  MACE_BENCHMARK(MACE_BM_PRELU_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_PRELU(N, C, H, W)              \
  MACE_BM_PRELU_MACRO(N, C, H, W, float, CPU); \
  MACE_BM_PRELU_MACRO(N, C, H, W, float, GPU); \
  MACE_BM_PRELU_MACRO(N, C, H, W, half, GPU);

MACE_BM_PRELU(1, 1, 512, 512);
MACE_BM_PRELU(1, 3, 128, 128);
MACE_BM_PRELU(1, 3, 512, 512);
MACE_BM_PRELU(1, 32, 112, 112);
MACE_BM_PRELU(1, 64, 256, 256);

namespace {
template <DeviceType D, typename T>
void TanhBenchmark(int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
  } else {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  }

  if (D == DeviceType::GPU) {
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
}  // namespace

#define MACE_BM_TANH_MACRO(N, C, H, W, TYPE, DEVICE)                         \
  static void MACE_BM_TANH_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(        \
      int iters) {                                                           \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;         \
    mace::testing::MaccProcessed(tot);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                      \
    TanhBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                          \
  }                                                                          \
  MACE_BENCHMARK(MACE_BM_TANH_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_TANH(N, C, H, W)              \
  MACE_BM_TANH_MACRO(N, C, H, W, float, CPU); \
  MACE_BM_TANH_MACRO(N, C, H, W, float, GPU); \
  MACE_BM_TANH_MACRO(N, C, H, W, half, GPU);

MACE_BM_TANH(1, 1, 512, 512);
MACE_BM_TANH(1, 3, 128, 128);
MACE_BM_TANH(1, 3, 512, 512);
MACE_BM_TANH(1, 32, 112, 112);
MACE_BM_TANH(1, 64, 256, 256);

namespace {
template <DeviceType D, typename T>
void SigmoidBenchmark(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
  } else {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  }

  if (D == DeviceType::GPU) {
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
}  // namespace

#define MACE_BM_SIGMOID_MACRO(N, C, H, W, TYPE, DEVICE)                   \
  static void MACE_BM_SIGMOID_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(  \
      int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;      \
    mace::testing::MaccProcessed(tot);                                    \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    SigmoidBenchmark<DEVICE, TYPE>(iters, N, C, H, W);                    \
  }                                                                       \
  MACE_BENCHMARK(MACE_BM_SIGMOID_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_SIGMOID(N, C, H, W)                 \
  MACE_BM_SIGMOID_MACRO(N, C, H, W, float, CPU);    \
  MACE_BM_SIGMOID_MACRO(N, C, H, W, float, GPU);    \
  MACE_BM_SIGMOID_MACRO(N, C, H, W, half, GPU);

MACE_BM_SIGMOID(1, 1, 512, 512);
MACE_BM_SIGMOID(1, 3, 128, 128);
MACE_BM_SIGMOID(1, 3, 512, 512);
MACE_BM_SIGMOID(1, 32, 112, 112);
MACE_BM_SIGMOID(1, 64, 256, 256);

}  // namespace test
}  // namespace ops
}  // namespace mace
