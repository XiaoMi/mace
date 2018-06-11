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

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void BMWinogradTransform(
    int iters, int batch, int height, int width, int channels, int block_size) {
  mace::testing::StopTiming();

  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  BufferToImage<D, T>(&net, "Input", "InputImage",
                      kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("WinogradTransform", "WinogradTransformTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .AddIntArg("block_size", block_size)
      .Finalize(net.NewOperatorDef());

  net.Setup(D);
  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.Run();
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_WINO_TRANSFORM_MACRO(N, H, W, C, M, TYPE, DEVICE)              \
  static void MACE_BM_WINO_TRANSFORM_##N##_##H##_##W##_##C##_##M##_##TYPE##_##\
    DEVICE( \
      int iters) {                                                             \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;           \
    mace::testing::MaccProcessed(tot);                                         \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    BMWinogradTransform<DEVICE, TYPE>(iters, N, H, W, C, M);                   \
  }                                                                            \
  MACE_BENCHMARK(                                                              \
    MACE_BM_WINO_TRANSFORM_##N##_##H##_##W##_##C##_##M##_##TYPE##_##DEVICE)

#define MACE_BM_WINO_TRANSFORM(N, H, W, C, M) \
  MACE_BM_WINO_TRANSFORM_MACRO(N, H, W, C, M, half, GPU);

MACE_BM_WINO_TRANSFORM(1, 128, 128, 3, 2);
MACE_BM_WINO_TRANSFORM(1, 256, 256, 3, 2);
MACE_BM_WINO_TRANSFORM(1, 64, 64, 3, 2);
MACE_BM_WINO_TRANSFORM(1, 128, 128, 3, 4);
MACE_BM_WINO_TRANSFORM(1, 256, 256, 3, 4);
MACE_BM_WINO_TRANSFORM(1, 64, 64, 3, 4);

namespace {
template <DeviceType D, typename T>
void BMWinogradInverseTransform(
    int iters, int batch, int height, int width, int channels, int block_size) {
  mace::testing::StopTiming();

  index_t p = batch * ((height + block_size - 1) / block_size) *
      ((width + block_size - 1) / block_size);
  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", {(block_size + 2) *
      (block_size + 2), channels, p, 1});

  BufferToImage<D, T>(&net, "Input", "InputImage",
                      kernels::BufferType::IN_OUT_HEIGHT);
  OpDefBuilder("WinogradInverseTransform", "WinogradInverseTransformTest")
      .Input("InputImage")
      .AddIntArg("batch", batch)
      .AddIntArg("height", height)
      .AddIntArg("width", width)
      .AddIntArg("block_size", block_size)
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  net.Setup(D);

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.Run();
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_WINO_INVERSE_TRANSFORM_MACRO(N, H, W, C, M, TYPE, DEVICE) \
  static void                                                             \
    MACE_BM_WINO_INVERSE_TRANSFORM_##N##_##H##_##W##_##C##_##M##_##TYPE##_\
    ##DEVICE(                                                             \
          int iters) {                                                    \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;      \
    mace::testing::MaccProcessed(tot);                                    \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    BMWinogradInverseTransform<DEVICE, TYPE>(iters, N, H, W, C, M);       \
  }                                                                       \
  MACE_BENCHMARK(                                                         \
  MACE_BM_WINO_INVERSE_TRANSFORM_##N##_##H##_##W##_##C##_##M##_##TYPE##_##\
  DEVICE)

#define MACE_BM_WINO_INVERSE_TRANSFORM(N, H, W, C, M) \
  MACE_BM_WINO_INVERSE_TRANSFORM_MACRO(N, H, W, C, M, half, GPU);

MACE_BM_WINO_INVERSE_TRANSFORM(1, 126, 126, 16, 2);
MACE_BM_WINO_INVERSE_TRANSFORM(1, 62, 62, 16, 2);
MACE_BM_WINO_INVERSE_TRANSFORM(1, 254, 254, 16, 2);

MACE_BM_WINO_INVERSE_TRANSFORM(1, 126, 126, 16, 4);
MACE_BM_WINO_INVERSE_TRANSFORM(1, 62, 62, 16, 4);
MACE_BM_WINO_INVERSE_TRANSFORM(1, 254, 254, 16, 4);

namespace {
template <DeviceType D, typename T>
void WinoFilterBufferToImage(int iters,
                         int out_channel, int in_channel,
                         int height, int width, int wino_block_size) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, T>("Input",
                           {out_channel, in_channel, height, width});

  OpDefBuilder("BufferToImage", "BufferToImageTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("buffer_type", kernels::BufferType::WINOGRAD_FILTER)
      .AddIntArg("wino_block_size", wino_block_size)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  net.Setup(D);
  for (int i = 0; i < 5; ++i) {
    net.Run();
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_WINO_B2I_MACRO(O, I, H, W, M, TYPE, DEVICE)                  \
  static void MACE_BM_WINO_B2I_##O##_##I##_##H##_##W##_##M##_##TYPE##_##DEVICE(\
      int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * O * I * H * W; \
    mace::testing::MaccProcessed(tot);                               \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    WinoFilterBufferToImage<DEVICE, TYPE>(iters, O, I, H, W, M);     \
  }                                                                  \
  MACE_BENCHMARK(\
  MACE_BM_WINO_B2I_##O##_##I##_##H##_##W##_##M##_##TYPE##_##DEVICE)

#define MACE_BM_WINO_B2I(O, I, H, W, M)              \
  MACE_BM_WINO_B2I_MACRO(O, I, H, W, M, half, GPU);

MACE_BM_WINO_B2I(16, 3, 3, 3, 2);
MACE_BM_WINO_B2I(16, 3, 3, 3, 4);
MACE_BM_WINO_B2I(32, 3, 3, 3, 2);
MACE_BM_WINO_B2I(32, 3, 3, 3, 4);
MACE_BM_WINO_B2I(128, 3, 3, 3, 2);
MACE_BM_WINO_B2I(128, 3, 3, 3, 4);
MACE_BM_WINO_B2I(256, 3, 3, 3, 2);
MACE_BM_WINO_B2I(256, 3, 3, 3, 4);

namespace {
template <DeviceType D, typename T>
void WinoMatMulBenchmark(
    int iters, int out_channels, int in_channels,
    int height, int width, int block_size) {
  mace::testing::StopTiming();

  OpsTestNet net;
  const int batch = (block_size + 2) * (block_size + 2);
  const index_t round_h = (height + block_size - 1) / block_size;
  const index_t round_w = (width + block_size - 1) / block_size;
  const index_t out_width = round_h * round_w;
  // Add input data
  net.AddRandomInput<D, float>("A", {batch, out_channels, in_channels, 1});
  net.AddRandomInput<D, float>("B", {batch, in_channels, out_width, 1});

  if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "A", "AImage", kernels::BufferType::IN_OUT_WIDTH);
    BufferToImage<D, T>(&net, "B", "BImage",
                        kernels::BufferType::IN_OUT_HEIGHT);

    OpDefBuilder("MatMul", "MatMulBM")
        .Input("AImage")
        .Input("BImage")
        .Output("Output")
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("MatMul", "MatMulBM")
        .Input("A")
        .Input("B")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
  }
  net.Setup(D);
  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.Run();
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_WINO_MATMUL_MACRO(OC, IC, H, W, M, TYPE, DEVICE)               \
  static void MACE_BM_WINO_MATMUL_##OC##_##IC##_##H##_##W##_##M##_##TYPE##_##\
    DEVICE(int iters) {                                                        \
    const int64_t macc = static_cast<int64_t>(iters) * OC * IC * H * W;        \
    const int64_t tot = static_cast<int64_t>(iters) * OC * (IC * H + H * W);   \
    mace::testing::MaccProcessed(macc);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    WinoMatMulBenchmark<DEVICE, TYPE>(iters, OC, IC, H, W, M);                 \
  }                                                                            \
  MACE_BENCHMARK(\
  MACE_BM_WINO_MATMUL_##OC##_##IC##_##H##_##W##_##M##_##TYPE##_##DEVICE)

#define MACE_BM_WINO_MATMUL(OC, IC, H, W, M)                 \
  MACE_BM_WINO_MATMUL_MACRO(OC, IC, H, W, M, half, GPU);

MACE_BM_WINO_MATMUL(16, 3, 128, 128, 2);
MACE_BM_WINO_MATMUL(16, 3, 128, 128, 4);
MACE_BM_WINO_MATMUL(32, 3, 256, 256, 2);
MACE_BM_WINO_MATMUL(32, 3, 256, 256, 4);

}  // namespace test
}  // namespace ops
}  // namespace mace
