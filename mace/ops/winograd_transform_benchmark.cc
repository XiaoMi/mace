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
    int iters, int batch, int height, int width, int channels) {
  mace::testing::StopTiming();

  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  BufferToImage<D, T>(&net, "Input", "InputImage",
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
}  // namespace

#define BM_WINOGRAD_TRANSFORM_MACRO(N, H, W, C, TYPE, DEVICE)                  \
  static void BM_WINOGRAD_TRANSFORM_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE( \
      int iters) {                                                             \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;           \
    mace::testing::MaccProcessed(tot);                                         \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    BMWinogradTransform<DEVICE, TYPE>(iters, N, H, W, C);                      \
  }                                                                            \
  BENCHMARK(BM_WINOGRAD_TRANSFORM_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE)

#define BM_WINOGRAD_TRANSFORM(N, H, W, C) \
  BM_WINOGRAD_TRANSFORM_MACRO(N, H, W, C, half, GPU);

BM_WINOGRAD_TRANSFORM(1, 16, 16, 128);
BM_WINOGRAD_TRANSFORM(1, 64, 64, 128);
BM_WINOGRAD_TRANSFORM(1, 128, 128, 128);

namespace {
template <DeviceType D, typename T>
void BMWinogradInverseTransform(
    int iters, int batch, int height, int width, int channels) {
  mace::testing::StopTiming();

  index_t p = batch * ((height + 1) / 2) * ((width + 1) / 2);
  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", {16, channels, p, 1});

  BufferToImage<D, T>(&net, "Input", "InputImage",
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
}  // namespace

#define BM_WINOGRAD_INVERSE_TRANSFORM_MACRO(N, H, W, C, TYPE, DEVICE)          \
  static void                                                                  \
      BM_WINOGRAD_INVERSE_TRANSFORM_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE( \
          int iters) {                                                         \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;           \
    mace::testing::MaccProcessed(tot);                                         \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    BMWinogradInverseTransform<DEVICE, TYPE>(iters, N, H, W, C);               \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_WINOGRAD_INVERSE_TRANSFORM_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE)

#define BM_WINOGRAD_INVERSE_TRANSFORM(N, H, W, C) \
  BM_WINOGRAD_INVERSE_TRANSFORM_MACRO(N, H, W, C, half, GPU);

BM_WINOGRAD_INVERSE_TRANSFORM(1, 14, 14, 32);
BM_WINOGRAD_INVERSE_TRANSFORM(1, 62, 62, 32);
BM_WINOGRAD_INVERSE_TRANSFORM(1, 126, 126, 32);

}  // namespace test
}  // namespace ops
}  // namespace mace
