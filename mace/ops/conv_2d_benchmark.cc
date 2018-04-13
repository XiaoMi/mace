//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

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
  if (D == DeviceType::NEON) {
    net.AddRandomInput<D, float>("Input", {batch, channels, height, width});
    net.AddRandomInput<D, float>("Filter",
                                 {output_channels, channels, kernel_h,
                                  kernel_w});
    net.AddRandomInput<D, float>("Bias", {output_channels});
  } else {
    net.AddRandomInput<D, float>("Input", {batch, height, width, channels});
    net.AddRandomInput<D, float>("Filter",
                                 {kernel_h, kernel_w, output_channels,
                                  channels});
    net.AddRandomInput<D, float>("Bias", {output_channels});
  }

  if (D == DeviceType::OPENCL) {
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
}  // namespace

// In common network, there are usually more than 1 layers, this is used to
// approximate the amortized latency. The OpenCL runtime for Mali/Adreno is
// in-order.

#define BM_CONV_2D_MACRO(N, C, H, W, KH, KW, STRIDE, DILATION, P, OC, TYPE,   \
                         DEVICE)                                              \
  static void                                                                 \
      BM_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##D##DILATION\
        ##_##P##_##OC##_##TYPE##_##DEVICE(                                    \
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
  BENCHMARK(                                                                  \
      BM_CONV_2D_##N##_##C##_##H##_##W##_K##KH##x##KW##S##STRIDE##D##DILATION\
        ##_##P##_##OC##_##TYPE##_##DEVICE)

#define BM_CONV_2D(N, C, H, W, KH, KW, S, D, P, OC)                 \
  BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, D, P, OC, float, CPU);    \
  BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, D, P, OC, float, NEON);   \
  BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, D, P, OC, float, OPENCL); \
  BM_CONV_2D_MACRO(N, C, H, W, KH, KW, S, D, P, OC, half, OPENCL);



// Filter sizes and data alignments
BM_CONV_2D(1, 64, 32, 32, 1, 1, 1, 1, VALID, 128);
BM_CONV_2D(1, 64, 33, 31, 1, 1, 1, 1, VALID, 128);
BM_CONV_2D(1, 64, 32, 32, 3, 3, 1, 1, SAME, 128);
BM_CONV_2D(1, 64, 33, 31, 3, 3, 1, 1, SAME, 128);
BM_CONV_2D(1, 64, 32, 32, 5, 5, 1, 1, SAME, 128);
BM_CONV_2D(1, 64, 32, 31, 5, 5, 1, 1, SAME, 128);
BM_CONV_2D(1, 64, 32, 31, 15, 1, 1, 1, SAME, 128);
BM_CONV_2D(1, 64, 32, 31, 1, 15, 1, 1, SAME, 128);

// 3 channels input
BM_CONV_2D(1, 3, 480, 480, 1, 1, 1, 1, VALID, 3);
BM_CONV_2D(1, 3, 224, 224, 3, 3, 2, 1, SAME, 32);
BM_CONV_2D(1, 3, 224, 224, 3, 3, 2, 1, VALID, 32);

// Dilations
BM_CONV_2D(1, 32, 256, 256, 3, 3, 1, 2, VALID, 32);
BM_CONV_2D(1, 32, 256, 256, 3, 3, 1, 4, VALID, 32);

// MobileNet
BM_CONV_2D(1, 128, 56, 56, 1, 1, 1, 1, SAME, 128);
BM_CONV_2D(1, 1024, 7, 7, 1, 1, 1, 1, SAME, 1024);

}  // namespace test
}  // namespace ops
}  // namespace mace
