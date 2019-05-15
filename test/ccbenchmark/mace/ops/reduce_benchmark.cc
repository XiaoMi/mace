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
void Reduce(int iters, int batch, int channels,
                int height, int width) {
  mace::testing::StopTiming();

  OpsTestNet net;
  // Add input data
  std::vector<int> axis = {1, 2};
  if (D == DeviceType::GPU) {
    net.AddRandomInput<D, T>("Input", {batch, height, width, channels});
  } else {
    net.AddRandomInput<D, T>("Input", {batch, channels, height, width});
  }

  OpDefBuilder("Reduce", "ReduceBM")
      .Input("Input")
      .AddIntsArg("axis", axis)
      .Output("OutputImage")
      .AddIntArg("has_data_format", 1)
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

#define MACE_BM_REDUCE_MACRO(N, C, H, W, TYPE, DEVICE)       \
  static void                                                \
    MACE_BM_REDUCE_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE(\
      int iters) {                                                   \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W; \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));              \
    Reduce<DEVICE, TYPE>(iters, N, C, H, W);        \
  }                                                                  \
  MACE_BENCHMARK(                                                         \
    MACE_BM_REDUCE_##N##_##C##_##H##_##W##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_REDUCE(N, C, H, W)                 \
  MACE_BM_REDUCE_MACRO(N, C, H, W, float, GPU);  \
  MACE_BM_REDUCE_MACRO(N, C, H, W, half, GPU);   \
  MACE_BM_REDUCE_MACRO(N, C, H, W, float, CPU)
#else
#define MACE_BM_REDUCE(N, C, H, W)                 \
  MACE_BM_REDUCE_MACRO(N, C, H, W, float, CPU)
#endif


MACE_BM_REDUCE(1, 1, 512, 512);
MACE_BM_REDUCE(4, 3, 128, 128);
MACE_BM_REDUCE(4, 1, 512, 512);
MACE_BM_REDUCE(16, 32, 112, 112);
MACE_BM_REDUCE(8, 64, 256, 256);
MACE_BM_REDUCE(1, 32, 480, 640);


}  // namespace test
}  // namespace ops
}  // namespace mace
