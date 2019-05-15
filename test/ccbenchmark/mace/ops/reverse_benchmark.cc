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
void Reverse(int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  net.AddRandomInput<D, T>("Input", {batch, channels, height, width});
  net.AddRandomInput<D, int32_t>("Axis", {1});

  OpDefBuilder("Reverse", "ReverseOpTest")
      .Input("Input")
      .Input("Axis")
      .Output("Output")
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

#define MACE_BM_REVERSE_MACRO(N, C, H, W, TYPE, DEVICE)                   \
  static void MACE_BM_REVERSE_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(  \
      int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;      \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    Reverse<DEVICE, TYPE>(iters, N, C, H, W);                             \
  }                                                                       \
  MACE_BENCHMARK(MACE_BM_REVERSE_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_REVERSE(N, C, H, W)                 \
  MACE_BM_REVERSE_MACRO(N, C, H, W, float, CPU);

MACE_BM_REVERSE(1, 1, 99, 256);
MACE_BM_REVERSE(1, 30, 99, 256);
MACE_BM_REVERSE(1, 50, 99, 256);
}  // namespace test
}  // namespace ops
}  // namespace mace
