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
void BMSpaceToBatch(
    int iters, int batch, int height, int width, int channels, int shape) {
  mace::testing::StopTiming();

  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", {batch, height, width, channels});

  BufferToImage<D, float>(&net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT_CHANNEL);
  OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntsArg("paddings", {shape, shape, shape, shape})
      .AddIntsArg("block_shape", {shape, shape})
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

#define BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, TYPE, DEVICE)             \
  static void                                                                \
      BM_SPACE_TO_BATCH_##N##_##H##_##W##_##C##_##SHAPE##_##TYPE##_##DEVICE( \
          int iters) {                                                       \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;         \
    mace::testing::MaccProcessed(tot);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                      \
    BMSpaceToBatch<DEVICE, TYPE>(iters, N, H, W, C, SHAPE);                  \
  }                                                                          \
  BENCHMARK(                                                                 \
      BM_SPACE_TO_BATCH_##N##_##H##_##W##_##C##_##SHAPE##_##TYPE##_##DEVICE)

#define BM_SPACE_TO_BATCH(N, H, W, C, SHAPE) \
  BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, float, GPU);

BM_SPACE_TO_BATCH(128, 16, 16, 128, 2);
BM_SPACE_TO_BATCH(1, 256, 256, 32, 2);
BM_SPACE_TO_BATCH(1, 256, 256, 32, 4);

}  // namespace test
}  // namespace ops
}  // namespace mace
