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
void BMTileHelper(int iters, const std::vector<index_t> &input_shape) {
  mace::testing::StopTiming();
  // Construct graph
  OpsTestNet net;
  net.AddRandomInput<D, T>("Input", input_shape);
  std::vector<int32_t> multiples = {};
  for (size_t i = 0; i < input_shape.size(); ++i) {
    multiples.push_back(2);
  }
  net.AddInputFromArray<D, int32_t>(
      "Multiples", {static_cast<int64_t>(multiples.size())}, multiples);

  OpDefBuilder("Tile", "TileBM")
      .Input("Input")
      .Input("Multiples")
      .Output("Output")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
    net.Sync();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
    net.Sync();
  }
}
}  // namespace

#define MACE_BM_TILE_MACRO(N, H, W, C, TYPE, DEVICE)                  \
  static void MACE_BM_TILE_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE( \
      int iters) {                                                    \
    const int64_t tot = static_cast<int64_t>(iters) * N * H * W * C;  \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));               \
    BMTileHelper<DEVICE, TYPE>(iters, {N, H, W, C});                  \
  }                                                                   \
  MACE_BENCHMARK(MACE_BM_TILE_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE)

#define MACE_BM_TILE(N, H, W, C) MACE_BM_TILE_MACRO(N, H, W, C, float, CPU);

MACE_BM_TILE(1, 32, 32, 5);
MACE_BM_TILE(1, 32, 32, 7);
MACE_BM_TILE(1, 32, 32, 3);
MACE_BM_TILE(1, 128, 128, 9);
MACE_BM_TILE(1, 128, 128, 7);

}  // namespace test
}  // namespace ops
}  // namespace mace
