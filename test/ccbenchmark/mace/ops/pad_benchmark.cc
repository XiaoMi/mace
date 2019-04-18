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
#include "mace/ops/pad.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void Pad(int iters, int batch, int height,
         int width, int channels, int pad, int pad_type) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, T>("Input", {batch, channels, height, width});
  } else {
    net.AddRandomInput<D, T>("Input", {batch, height, width, channels});
  }

  const std::vector<int> paddings = {0, 0, pad, pad, pad, pad, 0, 0};
  OpDefBuilder("Pad", "PadTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("paddings", paddings)
      .AddIntArg("pad_type", pad_type)
      .AddIntArg("has_data_format", 1)
      .AddFloatArg("constant_value", 1.0)
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

#define MACE_BM_PAD_MACRO(N, H, W, C, PAD, MODE, TYPE, DEVICE)               \
  static void MACE_BM_PAD_##N##_##H##_##W##_##C##_##PAD##_##MODE##_##TYPE    \
              ##_##DEVICE(                                                   \
      int iters) {                                                           \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;         \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                      \
    Pad<DEVICE, TYPE>(iters, N, H, W, C, PAD, MODE);                         \
  }                                                                          \
  MACE_BENCHMARK(MACE_BM_PAD_##N##_##H##_##W##_##C##_##PAD##_##MODE##_##TYPE \
                 ##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_PAD_MODE(N, H, W, C, PAD, MODE)            \
  MACE_BM_PAD_MACRO(N, H, W, C, PAD, MODE, float, CPU);    \
  MACE_BM_PAD_MACRO(N, H, W, C, PAD, MODE, float, GPU);    \
  MACE_BM_PAD_MACRO(N, H, W, C, PAD, MODE, half, GPU)
#else
#define MACE_BM_PAD_MODE(N, H, W, C, PAD, MODE)            \
  MACE_BM_PAD_MACRO(N, H, W, C, PAD, MODE, float, CPU)
#endif

#define MACE_BM_PAD(N, H, W, C, PAD)              \
  MACE_BM_PAD_MODE(N, H, W, C, PAD, CONSTANT);    \
  MACE_BM_PAD_MODE(N, H, W, C, PAD, REFLECT);     \
  MACE_BM_PAD_MODE(N, H, W, C, PAD, SYMMETRIC);

MACE_BM_PAD(1, 512, 512, 1, 2);
MACE_BM_PAD(1, 112, 112, 64, 1);
MACE_BM_PAD(1, 256, 256, 32, 2);
MACE_BM_PAD(1, 512, 512, 16, 2);

}  // namespace test
}  // namespace ops
}  // namespace mace
