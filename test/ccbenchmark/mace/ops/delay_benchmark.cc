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

template <DeviceType D, typename T>
static void Delay(int iters,
                      int batch,
                      int chunk,
                      int dim,
                      int offset) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, T>("Input", {batch, chunk, dim});

  OpDefBuilder("Delay", "DelayTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("offset", -offset)
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

#define MACE_BM_DELAY_MACRO(N, C, D, OFFSET, TYPE, DEVICE)                 \
  static void MACE_BM_DELAY_##N##_##C##_##D##_##OFFSET##_##TYPE##_##DEVICE(\
      int iters) {                                                             \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * D;               \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    Delay<DEVICE, TYPE>(iters, N, C, D, OFFSET);                           \
  }                                                                            \
  MACE_BENCHMARK(MACE_BM_DELAY_##N##_##C##_##D##_##OFFSET##_##TYPE\
##_##DEVICE)

#define MACE_BM_DELAY(N, C, D, OFFSET)                 \
  MACE_BM_DELAY_MACRO(N, C, D, OFFSET, float, CPU);

MACE_BM_DELAY(8, 40, 512, 2);
MACE_BM_DELAY(16, 80, 100, 3);
MACE_BM_DELAY(32, 60, 200, 5);


}  // namespace test
}  // namespace ops
}  // namespace mace
