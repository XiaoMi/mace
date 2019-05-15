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
void BMSpaceToBatch(
    int iters, int batch, int height, int width, int channels, int shape) {
  mace::testing::StopTiming();

  OpsTestNet net;
  if (D == DeviceType::CPU) {
    if (DataTypeToEnum<T>::value != DT_UINT8) {
      net.AddRandomInput<D, float>(
          "Input", {batch, channels, height, width});
    } else {
      net.AddRandomInput<DeviceType::CPU, uint8_t>(
          "Input", {batch, height, width, channels});
    }
  } else if (D == DeviceType::GPU) {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  }

  OpDefBuilder("SpaceToBatchND", "SpaceToBatchNDTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("paddings", {shape, shape, shape, shape})
      .AddIntsArg("block_shape", {shape, shape})
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

#define MACE_BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, TYPE, DEVICE)          \
  static void                                                                  \
    MACE_BM_SPACE_TO_BATCH_##N##_##H##_##W##_##C##_##SHAPE##_##TYPE##_##DEVICE(\
        int iters) {                                                           \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;           \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    BMSpaceToBatch<DEVICE, TYPE>(iters, N, H, W, C, SHAPE);                    \
  }                                                                            \
  MACE_BENCHMARK(                                                              \
    MACE_BM_SPACE_TO_BATCH_##N##_##H##_##W##_##C##_##SHAPE##_##TYPE##_##DEVICE)

#if defined(MACE_ENABLE_OPENCL) && defined(MACE_ENABLE_QUANTIZE)
#define MACE_BM_SPACE_TO_BATCH(N, H, W, C, SHAPE)              \
  MACE_BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, float, GPU); \
  MACE_BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, float, CPU); \
  MACE_BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, uint8_t, CPU)
#elif defined(MACE_ENABLE_OPENCL)
#define MACE_BM_SPACE_TO_BATCH(N, H, W, C, SHAPE)              \
  MACE_BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, float, GPU); \
  MACE_BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, float, CPU)
#elif defined(MACE_ENABLE_QUANTIZE)
#define MACE_BM_SPACE_TO_BATCH(N, H, W, C, SHAPE)              \
  MACE_BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, float, CPU); \
  MACE_BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, uint8_t, CPU)
#else
#define MACE_BM_SPACE_TO_BATCH(N, H, W, C, SHAPE)              \
  MACE_BM_SPACE_TO_BATCH_MACRO(N, H, W, C, SHAPE, float, CPU)
#endif

MACE_BM_SPACE_TO_BATCH(128, 16, 16, 128, 2);
MACE_BM_SPACE_TO_BATCH(1, 256, 256, 32, 2);
MACE_BM_SPACE_TO_BATCH(1, 256, 256, 16, 2);
MACE_BM_SPACE_TO_BATCH(1, 256, 256, 32, 4);
MACE_BM_SPACE_TO_BATCH(1, 256, 256, 32, 8);

}  // namespace test
}  // namespace ops
}  // namespace mace
