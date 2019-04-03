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

#include "mace/benchmark/statistics.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void ResizeBilinearBenchmark(int iters,
                             int batch,
                             int channels,
                             int input_height,
                             int input_width,
                             int output_height,
                             int output_width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    if (DataTypeToEnum<T>::value != DT_UINT8) {
      net.AddRandomInput<D, float>("Input",
                                   {batch, channels, input_height,
                                    input_width});
    } else {
      net.AddRandomInput<D, uint8_t>("Input",
                                   {batch, input_height, input_width,
                                    channels});
    }
  } else if (D == DeviceType::GPU) {
    net.AddRandomInput<D, float>("Input",
                                 {batch, input_height, input_width, channels});
  } else {
    MACE_NOT_IMPLEMENTED;
  }
  OpDefBuilder("ResizeBilinear", "ResizeBilinearBenchmark")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("size", {output_height, output_width})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.RunOp(D);
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.RunOp(D);
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, TYPE, DEVICE)      \
  static void                                                                  \
      MACE_BM_RESIZE_BILINEAR_##N##_##C##_##H0##_##W0##_##H1##_##W1##_##TYPE##_\
        ##DEVICE(                                                              \
          int iters) {                                                         \
    const int64_t macs = static_cast<int64_t>(iters) *                         \
        mace::benchmark::StatMACs("ResizeBilinear", {}, {N, H1, W1, C});       \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H0 * W0;         \
    mace::testing::MacsProcessed(macs);                                        \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                        \
    ResizeBilinearBenchmark<DEVICE, TYPE>(iters, N, C, H0, W0, H1, W1);        \
  }                                                                            \
  MACE_BENCHMARK(                                                              \
      MACE_BM_RESIZE_BILINEAR_##N##_##C##_##H0##_##W0##_##H1##_##W1##_##TYPE##_\
        ##DEVICE)

#if defined(MACE_ENABLE_OPENCL) && defined(MACE_ENABLE_QUANTIZE)
#define MACE_BM_RESIZE_BILINEAR(N, C, H0, W0, H1, W1)                 \
  MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, float, CPU);    \
  MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, uint8_t, CPU);    \
  MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, float, GPU);    \
  MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, half, GPU)
#elif defined(MACE_ENABLE_OPENCL)
#define MACE_BM_RESIZE_BILINEAR(N, C, H0, W0, H1, W1)                 \
  MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, float, CPU);    \
  MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, float, GPU);    \
  MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, half, GPU)
#elif defined(MACE_ENABLE_QUANTIZE)
#define MACE_BM_RESIZE_BILINEAR(N, C, H0, W0, H1, W1)                 \
  MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, float, CPU);    \
  MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, uint8_t, CPU)
#else
#define MACE_BM_RESIZE_BILINEAR(N, C, H0, W0, H1, W1)                 \
  MACE_BM_RESIZE_BILINEAR_MACRO(N, C, H0, W0, H1, W1, float, CPU)
#endif

MACE_BM_RESIZE_BILINEAR(1, 128, 120, 120, 480, 480);
MACE_BM_RESIZE_BILINEAR(1, 256, 7, 7, 15, 15);
MACE_BM_RESIZE_BILINEAR(1, 256, 15, 15, 30, 30);
MACE_BM_RESIZE_BILINEAR(1, 128, 30, 30, 60, 60);
MACE_BM_RESIZE_BILINEAR(1, 128, 240, 240, 480, 480);
MACE_BM_RESIZE_BILINEAR(1, 3, 4032, 3016, 480, 480);
MACE_BM_RESIZE_BILINEAR(1, 3, 480, 480, 4032, 3016);

}  // namespace test
}  // namespace ops
}  // namespace mace
