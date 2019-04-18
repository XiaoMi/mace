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

class CumsumOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void Cumsum(int iters, int batch, int channels, int height, int width) {
  mace::testing::StopTiming();

  // Construct graph
  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, T>("Input", {batch, channels, height, width});
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  OpDefBuilder("Cumsum", "CumsumTest")
    .Input("Input")
    .Output("Output")
    .AddIntArg("axis", 0)
    .AddIntArg("exclusive", 0)
    .AddIntArg("reverse", 0)
    .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
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

#define MACE_BM_CUMSUM_MACRO(N, C, H, W, TYPE, DEVICE)                  \
  static void MACE_BM_CUMSUM_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE( \
      int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;      \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                   \
    Cumsum<DEVICE, TYPE>(iters, N, C, H, W);                             \
  }                                                                       \
  MACE_BENCHMARK(MACE_BM_CUMSUM_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#define MACE_BM_CUMSUM(N, C, H, W)                 \
  MACE_BM_CUMSUM_MACRO(N, C, H, W, float, CPU);

MACE_BM_CUMSUM(1, 1, 512, 512);
MACE_BM_CUMSUM(1, 3, 128, 128);
MACE_BM_CUMSUM(1, 3, 512, 512);
MACE_BM_CUMSUM(1, 32, 112, 112);
MACE_BM_CUMSUM(1, 64, 256, 256);
MACE_BM_CUMSUM(1, 64, 512, 512);
MACE_BM_CUMSUM(1, 128, 56, 56);
MACE_BM_CUMSUM(1, 128, 256, 256);
MACE_BM_CUMSUM(1, 256, 14, 14);
MACE_BM_CUMSUM(1, 512, 14, 14);
MACE_BM_CUMSUM(1, 1024, 7, 7);
MACE_BM_CUMSUM(32, 1, 256, 256);
MACE_BM_CUMSUM(32, 3, 256, 256);

}  // namespace test
}  // namespace ops
}  // namespace mace
