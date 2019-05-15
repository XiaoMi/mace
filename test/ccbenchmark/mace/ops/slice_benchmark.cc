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
template<DeviceType D, typename T>
void BMSliceHelper(int iters,
                   const std::vector<index_t> &input_shape,
                   const int offset,
                   const int output_dim) {
  mace::testing::StopTiming();
  // Construct graph
  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", input_shape);
  OpDefBuilder("Slice", "SliceBM")
      .Input("Input")
      .Output("Output")
      .AddIntArg("offset", offset)
      .AddIntArg("output_dim", output_dim)
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

#define MACE_BM_SLICE_MACRO(N, H, W, S, D, TYPE, DEVICE)  \
  static void                                                                \
      MACE_BM_SLICE_##N##_##H##_##W##_##S##_##D##_##TYPE##_##DEVICE(  \
          int iters) {                                                       \
        const int64_t tot = static_cast<int64_t>(iters) * N * H * W;         \
        mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                  \
        BMSliceHelper<DEVICE, TYPE>(iters, {N, H, W}, S, D);             \
      }                                                                      \
      MACE_BENCHMARK(                                                        \
          MACE_BM_SLICE_##N##_##H##_##W##_##S##_##D##_##TYPE##_##DEVICE)

#define MACE_BM_SLICE(N, H, W, S, D)                 \
  MACE_BM_SLICE_MACRO(N, H, W, S, D, float, CPU);

MACE_BM_SLICE(1, 32, 32, 5, 5);
MACE_BM_SLICE(1, 32, 32, 7, 5);
MACE_BM_SLICE(1, 32, 32, 3, 20);
MACE_BM_SLICE(1, 128, 128, 9, 100);
MACE_BM_SLICE(1, 128, 128, 7, 100);

}  // namespace test
}  // namespace ops
}  // namespace mace
