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

#include "mace/kernels/global_avg_pooling.h"
#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D>
void GlobalAvgPooling(
    int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;
  OpDefBuilder("GlobalAvgPooling", "GlobalAvgPoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input",
                                             {batch, channels, height, width});

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
}
}  // namespace

#define BM_GLOBAL_AVG_POOLING_MACRO(N, C, H, W, DEVICE)               \
  static void BM_GLOBAL_AVG_POOLING_##N##_##C##_##H##_##W##_##DEVICE( \
      int iters) {                                                    \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;  \
    mace::testing::MaccProcessed(tot);                                \
    mace::testing::BytesProcessed(tot *(sizeof(float)));              \
    GlobalAvgPooling<DEVICE>(iters, N, C, H, W);                      \
  }                                                                   \
  BENCHMARK(BM_GLOBAL_AVG_POOLING_##N##_##C##_##H##_##W##_##DEVICE)

#define BM_GLOBAL_AVG_POOLING(N, C, H, W) \
  BM_GLOBAL_AVG_POOLING_MACRO(N, C, H, W, CPU);
//  BM_GLOBAL_AVG_POOLING_MACRO(N, C, H, W, NEON);

BM_GLOBAL_AVG_POOLING(1, 3, 7, 7);
BM_GLOBAL_AVG_POOLING(1, 3, 64, 64);
BM_GLOBAL_AVG_POOLING(1, 3, 256, 256);

}  // namespace test
}  // namespace ops
}  // namespace mace
