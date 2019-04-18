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
void OneHot(int iters, int batch, int depth, int axis) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, T>("Input", {batch});

  OpDefBuilder("OneHot", "OneHotTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("depth", depth)
      .AddIntArg("axis", axis)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }
  net.Sync();

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_ONE_HOT_MACRO(N, DEPTH, AXIS, TYPE, DEVICE)                  \
  static void MACE_BM_ONE_HOT_##N##_##DEPTH##_##AXIS##_##TYPE##_##DEVICE(    \
      int iters) {                                                           \
    const int64_t tot = static_cast<int64_t>(iters) * N;                     \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                      \
    OneHot<DEVICE, TYPE>(iters, N, DEPTH, AXIS);                             \
  }                                                                          \
  MACE_BENCHMARK(MACE_BM_ONE_HOT_##N##_##DEPTH##_##AXIS##_##TYPE##_##DEVICE)

#define MACE_BM_ONE_HOT(N, DEPTH, AXIS)                 \
  MACE_BM_ONE_HOT_MACRO(N, DEPTH, AXIS, float, CPU);

MACE_BM_ONE_HOT(512, 16, 0);
MACE_BM_ONE_HOT(512, 16, 1);
MACE_BM_ONE_HOT(5000, 5000, 0);
MACE_BM_ONE_HOT(5000, 5000, 1);
MACE_BM_ONE_HOT(15000, 500, 0);
MACE_BM_ONE_HOT(15000, 500, 1);
MACE_BM_ONE_HOT(15000, 5000, 0);
MACE_BM_ONE_HOT(15000, 5000, 1);

}  // namespace test
}  // namespace ops
}  // namespace mace
