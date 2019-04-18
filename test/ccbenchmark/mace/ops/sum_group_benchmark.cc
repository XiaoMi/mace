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
void SumGroupBenchmark(int iters, int n, int h, int w) {
  mace::testing::StopTiming();
  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, float>("Input", {n, h, w});
  net.AddRepeatedInput<D, int>("Sizes",
                               {w / 2},
                               2);
  OpDefBuilder("SumGroup", "SumGroupBM")
      .Input("Input")
      .Input("Sizes")
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

#define MACE_BM_SUMGROUP_MACRO(N, H, W, TYPE, DEVICE)            \
  static void                                                    \
      MACE_BM_SUMGROUP_##N##_##H##_##W##_##TYPE##_##DEVICE(      \
          int iters) {                                           \
    const int64_t tot = static_cast<int64_t>(iters) * N * H * W; \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));          \
    SumGroupBenchmark<DEVICE, TYPE>(iters, N, H, W);             \
  }                                                              \
  MACE_BENCHMARK(                                                \
      MACE_BM_SUMGROUP_##N##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_SUMGROUP(N, H, W)             \
  MACE_BM_SUMGROUP_MACRO(N, H, W, float, CPU);

MACE_BM_SUMGROUP(1, 10, 256);
MACE_BM_SUMGROUP(1, 20, 128);
MACE_BM_SUMGROUP(1, 10, 128);
MACE_BM_SUMGROUP(1, 20, 512);

}  // namespace test
}  // namespace ops
}  // namespace mace
