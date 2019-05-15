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

#include "mace/utils/statistics.h"
#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void ResizeNearestNeighborBenchmark(int iters,
                                    int batch,
                                    int channels,
                                    int input_height,
                                    int input_width,
                                    int output_height,
                                    int output_width) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  std::vector<int32_t> size = {output_height, output_width};
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input",
                                 {batch, channels, input_height, input_width});
    net.AddInputFromArray<D, int32_t>("Size", {2}, size);
  } else if (D == DeviceType::GPU) {
    net.AddRandomInput<D, float>("Input",
                                 {batch, input_height, input_width, channels});
    net.AddInputFromArray<D, int32_t>("Size", {2}, size);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  OpDefBuilder("ResizeNearestNeighbor", "ResizeNearestNeighborBenchmark")
      .Input("Input")
      .Input("Size")
      .Output("Output")
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

#define MACE_BM_RESIZE_NEAREST_NEIGHBOR_MACRO(N, C, H0, W0, H1, W1, TYPE,     \
                                              DEVICE)                         \
  static void                                                                 \
      MACE_BM_RESIZE_NEAREST_NEIGHBOR_##N##_##C##_##H0##_##W0##_##H1##_##W1##_\
        ##TYPE##_##DEVICE(                                                    \
          int iters) {                                                        \
    const int64_t macs = static_cast<int64_t>(iters) *                        \
        mace::benchmark::StatMACs("ResizeNearestNeighbor",                    \
          {}, {N, H1, W1, C});                                                \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H0 * W0;        \
    mace::testing::MacsProcessed(macs);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    ResizeNearestNeighborBenchmark<DEVICE, TYPE>(iters, N, C, H0, W0, H1, W1);\
  }                                                                           \
  MACE_BENCHMARK(                                                             \
      MACE_BM_RESIZE_NEAREST_NEIGHBOR_##N##_##C##_##H0##_##W0##_##H1##_##W1##_\
        ##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_RESIZE_NEAREST_NEIGHBOR(N, C, H0, W0, H1, W1)                 \
  MACE_BM_RESIZE_NEAREST_NEIGHBOR_MACRO(N, C, H0, W0, H1, W1, float, CPU);    \
  MACE_BM_RESIZE_NEAREST_NEIGHBOR_MACRO(N, C, H0, W0, H1, W1, float, GPU);    \
  MACE_BM_RESIZE_NEAREST_NEIGHBOR_MACRO(N, C, H0, W0, H1, W1, half, GPU)
#else
#define MACE_BM_RESIZE_NEAREST_NEIGHBOR(N, C, H0, W0, H1, W1)                 \
  MACE_BM_RESIZE_NEAREST_NEIGHBOR_MACRO(N, C, H0, W0, H1, W1, float, CPU)
#endif

MACE_BM_RESIZE_NEAREST_NEIGHBOR(1, 128, 120, 120, 480, 480);
MACE_BM_RESIZE_NEAREST_NEIGHBOR(1, 256, 7, 7, 15, 15);
MACE_BM_RESIZE_NEAREST_NEIGHBOR(1, 256, 15, 15, 30, 30);
MACE_BM_RESIZE_NEAREST_NEIGHBOR(1, 128, 30, 30, 60, 60);
MACE_BM_RESIZE_NEAREST_NEIGHBOR(1, 128, 240, 240, 480, 480);
MACE_BM_RESIZE_NEAREST_NEIGHBOR(1, 3, 4032, 3016, 480, 480);
MACE_BM_RESIZE_NEAREST_NEIGHBOR(1, 3, 480, 480, 4032, 3016);

}  // namespace test
}  // namespace ops
}  // namespace mace
