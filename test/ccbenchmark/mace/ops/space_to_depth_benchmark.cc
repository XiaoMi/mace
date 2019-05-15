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
void SpaceToDepth(
    int iters, int batch, int channels, int height, int width, int block_size) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
  } else if (D == DeviceType::GPU) {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  OpDefBuilder("SpaceToDepth", "SpaceToDepthBM")
      .Input("Input")
      .Output("Output")
      .AddIntArg("block_size", block_size)
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

#define MACE_BM_SPACE_TO_DEPTH_MACRO(N, C, H, W, G, TYPE, DEVICE)             \
  static void                                                                 \
      MACE_BM_SPACE_TO_DEPTH_##N##_##C##_##H##_##W##_##G##_##TYPE##_##DEVICE( \
          int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;          \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    SpaceToDepth<DEVICE, TYPE>(iters, N, C, H, W, G);                         \
  }                                                                           \
  MACE_BENCHMARK(                                                             \
      MACE_BM_SPACE_TO_DEPTH_##N##_##C##_##H##_##W##_##G##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_SPACE_TO_DEPTH(N, C, H, W, G)                 \
  MACE_BM_SPACE_TO_DEPTH_MACRO(N, C, H, W, G, float, CPU);    \
  MACE_BM_SPACE_TO_DEPTH_MACRO(N, C, H, W, G, float, GPU);    \
  MACE_BM_SPACE_TO_DEPTH_MACRO(N, C, H, W, G, half, GPU)
#else
#define MACE_BM_SPACE_TO_DEPTH(N, C, H, W, G)                 \
  MACE_BM_SPACE_TO_DEPTH_MACRO(N, C, H, W, G, float, CPU)
#endif

MACE_BM_SPACE_TO_DEPTH(1, 1, 513, 513, 3);
MACE_BM_SPACE_TO_DEPTH(1, 2, 256, 256, 2);
MACE_BM_SPACE_TO_DEPTH(1, 3, 512, 512, 2);
MACE_BM_SPACE_TO_DEPTH(1, 3, 513, 513, 3);
MACE_BM_SPACE_TO_DEPTH(1, 64, 64, 64, 4);
MACE_BM_SPACE_TO_DEPTH(1, 64, 128, 128, 4);
MACE_BM_SPACE_TO_DEPTH(1, 64, 256, 256, 4);

}  // namespace test
}  // namespace ops
}  // namespace mace
