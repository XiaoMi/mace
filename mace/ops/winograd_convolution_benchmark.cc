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

#include "mace/core/operator.h"
#include "mace/core/testing/test_benchmark.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

namespace {
template <DeviceType D, typename T>
void BMWinogradConvolution(
    int iters, int batch, int height, int width,
    int in_channels, int out_channels, int block_size) {
  mace::testing::StopTiming();
  OpsTestNet net;
  net.AddRandomInput<D, float>("Input", {batch, height, width, in_channels});

  net.AddRandomInput<D, float>("Filter", {out_channels, in_channels, 3, 3});
  net.AddRandomInput<D, T>("Bias", {out_channels});

  BufferToImage<D, T>(&net, "Input", "InputImage",
                      kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<D, T>(&net, "Filter", "FilterImage",
                      kernels::BufferType::CONV2D_FILTER);
  BufferToImage<D, T>(&net, "Bias", "BiasImage", kernels::BufferType::ARGUMENT);

  // Winograd convolution
  // transform filter
    BufferToImage<D, T>(&net, "Filter", "WinoFilter",
                        kernels::BufferType::WINOGRAD_FILTER, block_size);

  // Inference convolution output shape
  OpDefBuilder("InferConv2dShape", "InferConv2dShapeTest")
      .Input("InputImage")
      .Output("ShapeOutput")
      .AddIntArg("data_format", 0)
      .AddIntsArg("strides", {1, 1})
      .AddIntsArg("kernels", {static_cast<int>(out_channels),
                              static_cast<int>(in_channels),
                              3, 3})
      .AddIntArg("padding", Padding::SAME)
      .OutputType({DataTypeToEnum<int32_t>::v()})
      .Finalize(net.NewOperatorDef());

  // Transform input
  OpDefBuilder("WinogradTransform", "WinogradTransformTest")
      .Input("InputImage")
      .Output("WinoInput")
      .AddIntArg("padding", Padding::SAME)
      .AddIntArg("wino_block_size", block_size)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.AddNewOperatorDef());

  // MatMul
  OpDefBuilder("MatMul", "MatMulTest")
      .Input("WinoFilter")
      .Input("WinoInput")
      .Output("WinoGemm")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.AddNewOperatorDef());

  // Inverse transform
  OpDefBuilder("WinogradInverseTransform", "WinogradInverseTransformTest")
      .Input("WinoGemm")
      .Input("ShapeOutput")
      .Input("BiasImage")
      .AddIntArg("batch", batch)
      .AddIntArg("height", height)
      .AddIntArg("width", width)
      .AddIntArg("wino_block_size", block_size)
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.AddNewOperatorDef());
  net.Setup(D);
  // Warm-up
  for (int i = 0; i < 5; ++i) {
    net.Run();
  }
  net.Sync();
  mace::testing::StartTiming();
  while (iters--) {
    net.Run();
  }
  net.Sync();
}
}  // namespace

#define MACE_BM_WINOGRAD_CONV_MACRO(N, H, W, IC, OC, M, TYPE, DEVICE)          \
  static void MACE_BM_WINOGRAD_CONV_##N##_##H##_##W##_##IC##_##OC##_##M##_##\
    TYPE##_##DEVICE( \
      int iters) {                                                             \
    const int64_t tot = static_cast<int64_t>(iters) * N * IC * H * W;          \
    const int64_t macc =                                                      \
        static_cast<int64_t>(iters) * N * OC * H * W * (3 * 3 * IC + 1);      \
    mace::testing::MaccProcessed(macc);                                       \
    mace::testing::BytesProcessed(tot *(sizeof(TYPE)));                       \
    BMWinogradConvolution<DEVICE, TYPE>(iters, N, H, W, IC, OC, M);            \
  }                                                                            \
  MACE_BENCHMARK(                                                              \
  MACE_BM_WINOGRAD_CONV_##N##_##H##_##W##_##IC##_##OC##_##M##_##TYPE##_##DEVICE)

#define MACE_BM_WINOGRAD_CONV(N, H, W, IC, OC, M) \
  MACE_BM_WINOGRAD_CONV_MACRO(N, H, W, IC, OC, M, half, GPU);


MACE_BM_WINOGRAD_CONV(1, 64, 64, 3, 16, 2);
MACE_BM_WINOGRAD_CONV(1, 128, 128, 3, 16, 2);
MACE_BM_WINOGRAD_CONV(1, 256, 256, 3, 16, 2);
MACE_BM_WINOGRAD_CONV(1, 64, 64, 3, 16, 4);
MACE_BM_WINOGRAD_CONV(1, 128, 128, 3, 16, 4);
MACE_BM_WINOGRAD_CONV(1, 256, 256, 3, 16, 4);
MACE_BM_WINOGRAD_CONV(1, 28, 28, 256, 256, 2);
MACE_BM_WINOGRAD_CONV(1, 28, 28, 256, 256, 4);
MACE_BM_WINOGRAD_CONV(1, 56, 56, 256, 256, 2);
MACE_BM_WINOGRAD_CONV(1, 56, 56, 256, 256, 4);
MACE_BM_WINOGRAD_CONV(1, 128, 128, 128, 256, 2);
MACE_BM_WINOGRAD_CONV(1, 128, 128, 128, 256, 4);

}  // namespace test
}  // namespace ops
}  // namespace mace
