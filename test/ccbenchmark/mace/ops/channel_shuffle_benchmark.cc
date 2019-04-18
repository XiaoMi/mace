// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void ChannelShuffle(
    int iters, int batch, int channels, int height, int width, int group) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input", {batch, height, channels, width});
  } else if (D == DeviceType::GPU) {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  OpDefBuilder("ChannelShuffle", "ChannelShuffleTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("group", group)
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

#define MACE_BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, TYPE, DEVICE)             \
  static void                                                                  \
      MACE_BM_CHANNEL_SHUFFLE_##N##_##C##_##H##_##W##_##G##_##TYPE##_##DEVICE( \
          int iters) {                                                         \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;           \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    ChannelShuffle<DEVICE, TYPE>(iters, N, C, H, W, G);                        \
  }                                                                            \
  MACE_BENCHMARK(                                                              \
      MACE_BM_CHANNEL_SHUFFLE_##N##_##C##_##H##_##W##_##G##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_CHANNEL_SHUFFLE(N, C, H, W, G)                 \
  MACE_BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, float, CPU);    \
  MACE_BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, float, GPU);    \
  MACE_BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, half, GPU);
#else
#define MACE_BM_CHANNEL_SHUFFLE(N, C, H, W, G)                 \
  MACE_BM_CHANNEL_SHUFFLE_MACRO(N, C, H, W, G, float, CPU);
#endif

MACE_BM_CHANNEL_SHUFFLE(1, 64, 64, 64, 8);
MACE_BM_CHANNEL_SHUFFLE(1, 64, 128, 128, 8);
MACE_BM_CHANNEL_SHUFFLE(1, 64, 256, 256, 8);

}  // namespace test
}  // namespace ops
}  // namespace mace
