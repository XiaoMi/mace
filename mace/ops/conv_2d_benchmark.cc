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
#include "mace/ops/conv_2d.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void Conv2d(int iters,
            int batch,
            int channels,
            int height,
            int width,
            int kernel_h,
            int kernel_w,
            int stride,
            int dilation,
            Padding padding,
            int output_channels) {
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
  net.AddRandomInput<D, float>("Filter",
                               {output_channels, channels, kernel_h,
                                kernel_w});
  net.AddRandomInput<D, float>("Bias", {output_channels});

  if (D == DeviceType::CPU) {
    OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {stride, stride})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {dilation, dilation})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Filter", "FilterImage",
                        kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, T>(&net, "Bias", "BiasImage",
                        kernels::BufferType::ARGUMENT);
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("Output")
        .AddIntsArg("strides", {stride, stride})
        .AddIntArg("padding", padding)
        .AddIntsArg("dilations", {dilation, dilation})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
  } else {
    MACE_NOT_IMPLEMENTED;
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

template <>
void Conv2d<CPU, uint8_t>(int iters,
                          int batch,
                          int channels,
                          int height,
                          int width,
                          int kernel_h,
                          int kernel_w,
                          int stride,
                          int dilation,
                          Padding padding,
                          int output_channels) {
  mace::testing::StopTiming();
  if (dilation > 1) {
    LOG(WARNING) << "uint8_t benchmarking dilation = 1 instead.";
  }

  OpsTestNet net;

  // Add input data
  net.AddRandomInput<DeviceType::CPU, uint8_t>(
      "Input", {batch, height, width, channels});
  net.GetTensor("Input")->SetScale(0.1);
  net.AddRandomInput<DeviceType::CPU, uint8_t>(
      "Filter", {output_channels, kernel_h, kernel_w, channels});
  net.GetTensor("Filter")->SetScale(0.1);
  net.AddRandomInput<DeviceType::CPU, int32_t>("Bias", {output_channels});
  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {stride, stride})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());

  net.Setup(DeviceType::CPU);

  net.GetTensor("Output")->SetScale(0.1);

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

}  // namespace

// In common network, there are usually more than 1 layers, this is used to
// approximate the amortized latency. The OpenCL runtime for Mali/Adreno is
// in-order.

#define MACE_BM_CONV_2D_MACRO(                                                \
    N, C, H, W, KH, KW, STRIDE, DILATION, P, OC, TYPE, DEVICE)                \
  static void                                                                 \
      MACE_BM_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##D##\
        DILATION##_##P##_##OC##_##TYPE##_##DEVICE(                            \
          int iters) {                                                        \
    const int64_t tot = static_cast<int64_t>(iters) * N * C * H * W;          \
    int64_t pad_h = 0, pad_w = 0;                                             \
    if (P == SAME) {                                                          \
      pad_h = KH / 2;                                                         \
      pad_w = KW / 2;                                                         \
    }                                                                         \
    int64_t oh =                                                              \
        (H + 2 * pad_h - KH - (KH - 1) * (DILATION - 1)) / STRIDE + 1;        \
    int64_t ow =                                                              \
        (W + 2 * pad_w - KW - (KW - 1) * (DILATION - 1)) / STRIDE + 1;        \
    const int64_t macc =                                                      \
        static_cast<int64_t>(iters) * N * OC * oh * ow * (KH * KW * C + 1);   \
    mace::testing::MaccProcessed(macc);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    Conv2d<DEVICE, TYPE>(iters, N, C, H, W, KH, KW, STRIDE, DILATION,         \
                         mace::Padding::P, OC);                               \
  }                                                                           \
  MACE_BENCHMARK(                                                             \
      MACE_BM_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##D##\
        DILATION##_##P##_##OC##_##TYPE##_##DEVICE)

#define MACE_BM_CONV_2D(N, C, H, W, KH, KW, S, D, P, OC)                 \
  MACE_BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, D, P, OC, float, CPU);    \
  MACE_BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, D, P, OC, float, GPU);    \
  MACE_BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, D, P, OC, half, GPU);     \
  MACE_BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, D, P, OC, uint8_t, CPU);



// Filter sizes and data alignments
MACE_BM_CONV_2D(1, 64, 32, 32, 1, 1, 1, 1, VALID, 128);
MACE_BM_CONV_2D(1, 64, 33, 31, 1, 1, 1, 1, VALID, 128);
MACE_BM_CONV_2D(1, 64, 32, 32, 3, 3, 1, 1, SAME, 128);
MACE_BM_CONV_2D(1, 64, 33, 31, 3, 3, 1, 1, SAME, 128);
MACE_BM_CONV_2D(1, 64, 32, 32, 5, 5, 1, 1, SAME, 128);
MACE_BM_CONV_2D(1, 64, 32, 31, 5, 5, 1, 1, SAME, 128);
MACE_BM_CONV_2D(1, 64, 32, 31, 15, 1, 1, 1, SAME, 128);
MACE_BM_CONV_2D(1, 64, 32, 31, 1, 15, 1, 1, SAME, 128);
MACE_BM_CONV_2D(1, 64, 32, 31, 7, 7, 1, 1, SAME, 128);
MACE_BM_CONV_2D(1, 64, 32, 31, 7, 7, 2, 1, SAME, 128);
MACE_BM_CONV_2D(1, 64, 32, 31, 7, 7, 3, 1, SAME, 128);

// 3 channels input
MACE_BM_CONV_2D(1, 3, 480, 480, 1, 1, 1, 1, VALID, 3);
MACE_BM_CONV_2D(1, 3, 224, 224, 3, 3, 2, 1, SAME, 32);
MACE_BM_CONV_2D(1, 3, 224, 224, 3, 3, 2, 1, VALID, 32);

// Dilations
MACE_BM_CONV_2D(1, 32, 256, 256, 3, 3, 1, 2, VALID, 32);
MACE_BM_CONV_2D(1, 32, 256, 256, 3, 3, 1, 4, VALID, 32);

// MobileNet
MACE_BM_CONV_2D(1, 128, 56, 56, 1, 1, 1, 1, SAME, 128);
MACE_BM_CONV_2D(1, 1024, 7, 7, 1, 1, 1, 1, SAME, 1024);

MACE_BM_CONV_2D(64, 32, 34, 34, 3, 3, 1, 1, VALID, 32);
MACE_BM_CONV_2D(1, 32, 34, 34, 3, 3, 1, 1, VALID, 32);

MACE_BM_CONV_2D(1, 192, 17, 17, 1, 7, 1, 1, SAME, 192);
MACE_BM_CONV_2D(1, 192, 17, 17, 7, 1, 1, 1, SAME, 192);
MACE_BM_CONV_2D(1, 160, 17, 17, 7, 1, 1, 1, SAME, 192);

MACE_BM_CONV_2D(1, 32, 256, 256, 1, 15, 1, 1, SAME, 2);
MACE_BM_CONV_2D(1, 32, 256, 256, 15, 1, 1, 1, SAME, 2);
MACE_BM_CONV_2D(1, 64, 64, 64, 15, 1, 1, 1, SAME, 2);

MACE_BM_CONV_2D(1, 3, 128, 128, 3, 3, 1, 1, SAME, 16);
MACE_BM_CONV_2D(1, 3, 256, 256, 3, 3, 1, 1, SAME, 16);
MACE_BM_CONV_2D(1, 3, 64, 64, 3, 3, 1, 1, SAME, 16);

}  // namespace test
}  // namespace ops
}  // namespace mace
