// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/ops/deconv_2d.h"
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
                                kernel_w});
  if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Filter", "FilterImage",
                        kernels::BufferType::CONV2D_FILTER);
    OpDefBuilder("Deconv2D", "Deconv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Output("Output")
        .AddIntsArg("strides", {stride, stride})
        .AddIntArg("padding", padding)
        .AddIntsArg("output_shape", {batch, out_h, out_w, output_channels})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    OpDefBuilder("Deconv2D", "Deconv2dTest")
        .Input("Input")
        .Input("Filter")
        .Output("Output")
        .AddIntsArg("strides", {stride, stride})
        .AddIntArg("padding", padding)
        .AddIntsArg("output_shape", {batch, out_h, out_w, output_channels})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  }

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
    int64_t oh = OH;                                                          \
    int64_t ow = OW;                                                          \
    const int64_t macc =                                                      \
        static_cast<int64_t>(iters) * N * OC * oh * ow * (KH * KW * C + 1);   \
    mace::testing::MaccProcessed(macc);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    Deconv2d<DEVICE, TYPE>(iters, N, C, H, W, KH, KW, STRIDE, OH, OW,         \
                         mace::Padding::P, OC);                               \
  }                                                                           \
  MACE_BENCHMARK(                                                             \
    MACE_BM_DECONV_2D_##N##_##C##_##H##_##W##_##KH##_##KW##_##STRIDE##_##OH##_\
        ##OW##_##P##_##OC##_##TYPE##_##DEVICE)

// TODO(liutuo): add cpu benchmark when optimized.
#define MACE_BM_DECONV_2D(N, C, H, W, KH, KW, S, OH, OW, P, OC)              \
  MACE_BM_DECONV_2D_MACRO(N, C, H, W, KH, KW, S, OH, OW, P, OC, float, GPU); \
  MACE_BM_DECONV_2D_MACRO(N, C, H, W, KH, KW, S, OH, OW, P, OC, half, GPU);

MACE_BM_DECONV_2D(1, 128, 15, 15, 1, 1, 1, 15, 15, VALID, 256);
MACE_BM_DECONV_2D(1, 32, 60, 60, 1, 1, 1, 60, 60, VALID, 128);

MACE_BM_DECONV_2D(1, 128, 60, 60, 3, 3, 1, 62, 62, VALID, 128);
MACE_BM_DECONV_2D(1, 32, 60, 60, 3, 3, 1, 60, 60, SAME, 32);
MACE_BM_DECONV_2D(1, 3, 512, 512, 7, 7, 2, 1023, 1023, SAME, 32);
MACE_BM_DECONV_2D(1, 128, 16, 16, 5, 5, 1, 20, 20, VALID, 32);
MACE_BM_DECONV_2D(1, 128, 64, 64, 5, 5, 1, 68, 68, VALID, 32);

MACE_BM_DECONV_2D(1, 3, 480, 480, 1, 1, 1, 480, 480, VALID, 3);

MACE_BM_DECONV_2D(1, 64, 32, 32, 1, 1, 1, 32, 32, VALID, 128);
MACE_BM_DECONV_2D(1, 64, 33, 32, 3, 3, 2, 65, 63, SAME, 128);
MACE_BM_DECONV_2D(1, 3, 224, 224, 3, 3, 2, 447, 447, SAME, 32);
MACE_BM_DECONV_2D(1, 3, 224, 224, 3, 3, 2, 449, 449, VALID, 32);

}  // namespace test
}  // namespace ops
}  // namespace mace
