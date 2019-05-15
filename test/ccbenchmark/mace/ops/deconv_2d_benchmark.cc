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
#include "mace/benchmark_utils/test_benchmark.h"
#include "mace/ops/common/conv_pool_2d_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

template <DeviceType D, typename T>
static void Deconv2d(int iters,
                   int batch,
                   int channels,
                   int height,
                   int width,
                   int kernel_h,
                   int kernel_w,
                   int stride,
                   int out_h,
                   int out_w,
                   Padding padding,
                   int output_channels) {
  mace::testing::StopTiming();

  OpsTestNet net;

  // Add input data
  if (D == DeviceType::CPU) {
    net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
  } else {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
  }
  net.AddRandomInput<D, float>("Filter",
                               {output_channels, channels, kernel_h,
                                kernel_w}, true);
  net.AddRandomInput<D, float>("Bias", {output_channels}, true);
  net.AddInputFromArray<D, int32_t>("OutputShape", {4},
                                    {batch, out_h, out_w, output_channels},
                                    true);
  OpDefBuilder("Deconv2D", "Deconv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("OutputShape")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {stride, stride})
      .AddIntArg("padding", padding)
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

#define MACE_BM_DECONV_2D_MACRO(                                              \
    N, C, H, W, KH, KW, STRIDE, OH, OW, P, OC, TYPE, DEVICE)                  \
  static void                                                                 \
    MACE_BM_DECONV_2D_##N##_##C##_##H##_##W##_##KH##_##KW##_##STRIDE##_##OH##_\
        ##OW##_##P##_##OC##_##TYPE##_##DEVICE(                                \
          int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;          \
    const int64_t macs =                                                      \
        static_cast<int64_t>(iters) * mace::benchmark::StatMACs(              \
            "Deconv2D", {OC, C, KH, KW}, {N, OH, OW, OC});                    \
    mace::testing::MacsProcessed(macs);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    Deconv2d<DEVICE, TYPE>(iters, N, C, H, W, KH, KW, STRIDE, OH, OW,         \
                         mace::Padding::P, OC);                               \
  }                                                                           \
  MACE_BENCHMARK(                                                             \
    MACE_BM_DECONV_2D_##N##_##C##_##H##_##W##_##KH##_##KW##_##STRIDE##_##OH##_\
        ##OW##_##P##_##OC##_##TYPE##_##DEVICE)

// TODO(liutuo): add cpu benchmark when optimized.
#ifdef MACE_ENABLE_OPENCL
#define MACE_BM_DECONV_2D(N, C, H, W, KH, KW, S, OH, OW, P, OC)              \
  MACE_BM_DECONV_2D_MACRO(N, C, H, W, KH, KW, S, OH, OW, P, OC, float, CPU); \
  MACE_BM_DECONV_2D_MACRO(N, C, H, W, KH, KW, S, OH, OW, P, OC, float, GPU); \
  MACE_BM_DECONV_2D_MACRO(N, C, H, W, KH, KW, S, OH, OW, P, OC, half, GPU)
#else
#define MACE_BM_DECONV_2D(N, C, H, W, KH, KW, S, OH, OW, P, OC)              \
  MACE_BM_DECONV_2D_MACRO(N, C, H, W, KH, KW, S, OH, OW, P, OC, float, CPU)
#endif

MACE_BM_DECONV_2D(1, 32, 60, 60, 1, 1, 1, 60, 60, VALID, 128);

MACE_BM_DECONV_2D(1, 128, 60, 60, 3, 3, 1, 62, 62, VALID, 128);
MACE_BM_DECONV_2D(1, 32, 60, 60, 3, 3, 1, 60, 60, SAME, 32);

MACE_BM_DECONV_2D(1, 32, 60, 60, 4, 4, 1, 60, 60, SAME, 32);
MACE_BM_DECONV_2D(1, 3, 224, 224, 4, 4, 2, 450, 450, VALID, 32);
MACE_BM_DECONV_2D(1, 3, 512, 512, 7, 7, 2, 1023, 1023, SAME, 32);
MACE_BM_DECONV_2D(1, 128, 16, 16, 5, 5, 1, 20, 20, VALID, 32);
MACE_BM_DECONV_2D(1, 128, 64, 64, 5, 5, 1, 68, 68, VALID, 32);

MACE_BM_DECONV_2D(1, 3, 480, 480, 1, 1, 1, 480, 480, VALID, 3);

MACE_BM_DECONV_2D(1, 64, 32, 32, 1, 1, 1, 32, 32, VALID, 128);
MACE_BM_DECONV_2D(1, 64, 33, 32, 3, 3, 2, 65, 63, SAME, 128);
MACE_BM_DECONV_2D(1, 3, 224, 224, 3, 3, 2, 448, 448, SAME, 32);

MACE_BM_DECONV_2D(1, 32, 1014, 762, 9, 9, 2, 2035, 1531, VALID, 1);

}  // namespace test
}  // namespace ops
}  // namespace mace
