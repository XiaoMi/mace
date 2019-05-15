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
void AddNBenchmark(int iters, int inputs, int n, int h, int w, int c) {
  mace::testing::StopTiming();

  OpsTestNet net;
  // Add input data
  for (int i = 0; i < inputs; ++i) {
    net.AddRandomInput<D, float>(MakeString("Input", i).c_str(), {n, h, w, c});
  }

  OpDefBuilder op_def_builder("AddN", "AddNBM");
  for (int i = 0; i < inputs; ++i) {
    op_def_builder.Input(MakeString("Input", i).c_str());
  }
  op_def_builder.Output("Output")
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

#define MACE_BM_ADDN_MACRO(INPUTS, N, H, W, C, TYPE, DEVICE)                  \
  static void                                                                 \
      MACE_BM_ADDN_##INPUTS##_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE(      \
          int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * INPUTS * N * H * W * C; \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    AddNBenchmark<DEVICE, TYPE>(iters, INPUTS, N, H, W, C);                   \
  }                                                                           \
  MACE_BENCHMARK(                                                             \
      MACE_BM_ADDN_##INPUTS##_##N##_##H##_##W##_##C##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_ADDN(INPUTS, N, H, W, C)                 \
  MACE_BM_ADDN_MACRO(INPUTS, N, H, W, C, float, CPU);    \
  MACE_BM_ADDN_MACRO(INPUTS, N, H, W, C, float, GPU);    \
  MACE_BM_ADDN_MACRO(INPUTS, N, H, W, C, half, GPU);
#else
#define MACE_BM_ADDN(INPUTS, N, H, W, C)                 \
  MACE_BM_ADDN_MACRO(INPUTS, N, H, W, C, float, CPU);
#endif

MACE_BM_ADDN(2, 1, 256, 256, 32);
MACE_BM_ADDN(2, 1, 128, 128, 32);
MACE_BM_ADDN(4, 1, 128, 128, 3);
MACE_BM_ADDN(2, 1, 256, 256, 3);
MACE_BM_ADDN(2, 1, 512, 512, 3);

}  // namespace test
}  // namespace ops
}  // namespace mace
