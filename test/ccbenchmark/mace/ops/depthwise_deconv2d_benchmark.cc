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

#include <algorithm>

#include "mace/utils/statistics.h"
#include "mace/core/operator.h"
#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

template <DeviceType D, typename T>
static void DepthwiseDeconv2d(int iters,
                              int batch,
                              int channels,
                              int height,
                              int width,
                              int kernel_h,
                              int kernel_w,
                              int stride,
                              int padding) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
  } else {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  }
  net.AddRandomInput<D, float>("Filter",
                               {1, channels, kernel_h,
                                kernel_w}, true);
  OpDefBuilder("DepthwiseDeconv2d", "DepthwiseDeconv2dTest")
      .Input("Input")
      .Input("Filter")
      .Output("Output")
      .AddIntsArg("strides", {stride, stride})
      .AddIntsArg("padding_values", {padding, padding})
      .AddIntArg("group", channels)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  net.Setup(D);

  // Warm-up
  for (int i = 0; i < 2; ++i) {
    net.Run();
    net.Sync();
  }

  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
    net.Sync();
  }
}

// In common network, there are usually more than 1 layers, this is used to
// approximate the amortized latency. The OpenCL runtime for Mali/Adreno is
// in-order.

#define MACE_BM_DEPTHWISE_DECONV2D_MACRO(                    \
    N, C, H, W, KH, KW, S, P, TYPE, DEVICE)                  \
  static void                                                                 \
    MACE_BM_DEPTHWISE_DECONV2D_##N##_##C##_##H##_##W##_##KH##_##KW##_##S##_##P\
        ##_##TYPE##_##DEVICE(                                                 \
          int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;          \
    const int64_t macs =                                                      \
        static_cast<int64_t>(iters) * mace::benchmark::StatMACs(              \
            "DepthwiseDeconv2d", {1, C, KH, KW}, {N, H, W, C});               \
    mace::testing::MacsProcessed(macs);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    DepthwiseDeconv2d<DEVICE, TYPE>(iters, N, C, H, W, KH, KW, S, P);         \
  }                                                                           \
  MACE_BENCHMARK(                                                             \
    MACE_BM_DEPTHWISE_DECONV2D_##N##_##C##_##H##_##W##_##KH##_##KW##_##S##_##P\
        ##_##TYPE##_##DEVICE)

#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_DEPTHWISE_DECONV2D(N, C, H, W, KH, KW, S, P)              \
  MACE_BM_DEPTHWISE_DECONV2D_MACRO(N, C, H, W, KH, KW, S, P, float, CPU); \
  MACE_BM_DEPTHWISE_DECONV2D_MACRO(N, C, H, W, KH, KW, S, P, float, GPU); \
  MACE_BM_DEPTHWISE_DECONV2D_MACRO(N, C, H, W, KH, KW, S, P, half, GPU);
#else
#define MACE_BM_DEPTHWISE_DECONV2D(N, C, H, W, KH, KW, S, P)              \
  MACE_BM_DEPTHWISE_DECONV2D_MACRO(N, C, H, W, KH, KW, S, P, float, CPU)
#endif

MACE_BM_DEPTHWISE_DECONV2D(1, 128, 15, 15, 1, 1, 1, 0);
MACE_BM_DEPTHWISE_DECONV2D(1, 32, 60, 60, 1, 1, 1, 0);

MACE_BM_DEPTHWISE_DECONV2D(1, 32, 60, 60, 3, 3, 1, 0);

MACE_BM_DEPTHWISE_DECONV2D(1, 128, 60, 60, 4, 4, 1, 0);
MACE_BM_DEPTHWISE_DECONV2D(1, 3, 224, 224, 4, 4, 2, 0);
MACE_BM_DEPTHWISE_DECONV2D(1, 3, 512, 512, 7, 7, 2, 0);
MACE_BM_DEPTHWISE_DECONV2D(1, 128, 16, 16, 5, 5, 1, 0);

MACE_BM_DEPTHWISE_DECONV2D(1, 64, 32, 32, 1, 1, 1, 0);
MACE_BM_DEPTHWISE_DECONV2D(1, 64, 33, 32, 3, 3, 2, 0);
MACE_BM_DEPTHWISE_DECONV2D(1, 3, 224, 224, 3, 3, 2, 0);
}  // namespace test
}  // namespace ops
}  // namespace mace
