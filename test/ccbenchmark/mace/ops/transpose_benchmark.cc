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
#include <vector>

#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template<DeviceType D, typename T>
void TransposeBenchmark(int iters,
                        std::vector<index_t> shape,
                        std::vector<int> dims) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", shape);

  OpDefBuilder("Transpose", "TransposeBM")
    .Input("Input")
    .Output("Output")
    .AddIntsArg("dims", dims)
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

#define MACE_BM_TRANSPOSE2D_MACRO(H, W, TYPE, DEVICE)                \
  static void MACE_BM_TRANSPOSE2D_##H##_##W##_##TYPE##_##DEVICE(     \
      int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * H * W;         \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    TransposeBenchmark<DEVICE, TYPE>(iters, {H, W}, {1, 0});         \
  }                                                                  \
  MACE_BENCHMARK(MACE_BM_TRANSPOSE2D_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_TRANSPOSE2D(H, W)                                    \
  MACE_BM_TRANSPOSE2D_MACRO(H, W, float, CPU);

#define MACE_BM_TRANSPOSE4D_MACRO(N, C, H, W, D0, D1, D2, D3, TYPE, DEVICE)   \
  static void                                                                 \
    MACE_BM_TRANSPOSE4D_##N##_##C##_##H##_##W##_##D0##D1##D2##D3##_##TYPE##_##\
      DEVICE(int iters) {                                                     \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;          \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    TransposeBenchmark<DEVICE, TYPE>(iters, {N, C, H, W}, {D0, D1, D2, D3});  \
  }                                                                           \
  MACE_BENCHMARK(                                                             \
    MACE_BM_TRANSPOSE4D_##N##_##C##_##H##_##W##_##D0##D1##D2##D3##_##TYPE##_##\
      DEVICE)

#define MACE_BM_TRANSPOSE4D(N, C, H, W, D0, D1, D2, D3)                   \
  MACE_BM_TRANSPOSE4D_MACRO(N, C, H, W, D0, D1, D2, D3, float, CPU);

MACE_BM_TRANSPOSE4D(1, 512, 512, 3, 0, 3, 1, 2);
MACE_BM_TRANSPOSE4D(1, 2, 512, 512, 0, 2, 3, 1);
MACE_BM_TRANSPOSE4D(1, 64, 64, 512, 0, 3, 1, 2);
MACE_BM_TRANSPOSE4D(1, 512, 64, 64, 0, 2, 3, 1);
MACE_BM_TRANSPOSE4D(1, 4, 20, 64, 0, 2, 1, 3);
MACE_BM_TRANSPOSE2D(128, 128);
MACE_BM_TRANSPOSE2D(512, 512);
MACE_BM_TRANSPOSE2D(1024, 1024);
MACE_BM_TRANSPOSE2D(512, 2048);
MACE_BM_TRANSPOSE2D(2048, 512);

}  // namespace test
}  // namespace ops
}  // namespace mace
