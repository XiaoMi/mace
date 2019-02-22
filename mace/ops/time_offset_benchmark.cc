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

#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template<DeviceType D, typename T>
void TimeOffsetBenchmark(int iters,
                         std::vector<index_t> shape,
                         int offset) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<D, float>("Input", shape);

  OpDefBuilder("TimeOffset", "TimeOffsetBM")
    .Input("Input")
    .Output("Output")
    .AddIntArg("offset", offset)
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

#define MACE_BM_TIMEOFFSET2D_MACRO(H, W, TYPE, DEVICE)              \
  static void MACE_BM_TIMEOFFSET2D_##H##_##W##_##TYPE##_##DEVICE(\
      int iters) {                                                     \
    const int64_t tot = static_cast<int64_t>(iters) * H * W;           \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                \
    TimeOffsetBenchmark<DEVICE, TYPE>(iters, {H, W}, 1);               \
  }                                                                    \
  MACE_BENCHMARK(MACE_BM_TIMEOFFSET2D_##H##_##W##_##TYPE##_##DEVICE)   \

#define MACE_BM_TIMEOFFSET2D(H, W)                           \
  MACE_BM_TIMEOFFSET2D_MACRO(H, W, float, CPU);


MACE_BM_TIMEOFFSET2D(20, 128);
MACE_BM_TIMEOFFSET2D(40, 512);
MACE_BM_TIMEOFFSET2D(1, 1024);
MACE_BM_TIMEOFFSET2D(20, 2048);
MACE_BM_TIMEOFFSET2D(20, 512);

}  // namespace test
}  // namespace ops
}  // namespace mace
