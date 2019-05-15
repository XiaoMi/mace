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

#include <string>

#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void PNormBenchmark(int iters, int n, int h, int w, int p, int ow) {
  mace::testing::StopTiming();

  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, float>("Input", {n, h, w});

  OpDefBuilder("PNorm", "PNormBM")
      .Input("Input")
      .AddIntArg("p", p)
      .AddIntArg("output_dim", ow)
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
    net.Sync();
  }
}
}  // namespace

#define MACE_BM_PNORM_MACRO(N, H, W, P, OW, TYPE, DEVICE)  \
  static void                                                    \
      MACE_BM_PNORM_##N##_##H##_##W##_##P##_##OW##_##TYPE##_##DEVICE( \
          int iters) {                                           \
    const int64_t tot = static_cast<int64_t>(iters) * N * H * W; \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));          \
    PNormBenchmark<DEVICE, TYPE>(iters, N, H, W, P, OW);         \
  }                                                              \
  MACE_BENCHMARK(                                                \
      MACE_BM_PNORM_##N##_##H##_##W##_##P##_##OW##_##TYPE##_##DEVICE)

#define MACE_BM_PNORM(N, H, W, P, OW)             \
  MACE_BM_PNORM_MACRO(N, H, W, P, OW, float, CPU);

MACE_BM_PNORM(1, 10, 256, 0, 128);
MACE_BM_PNORM(1, 20, 128, 1, 64);
MACE_BM_PNORM(1, 10, 128, 2, 64);
MACE_BM_PNORM(1, 16, 256, 0, 128);
MACE_BM_PNORM(1, 32, 128, 1, 64);
MACE_BM_PNORM(1, 10, 512, 2, 256);

}  // namespace test
}  // namespace ops
}  // namespace mace
