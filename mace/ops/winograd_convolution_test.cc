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

#include <fstream>

#include "mace/ops/conv_pool_2d_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class WinogradConvolutionTest : public OpsTestBase {};

namespace {

template <DeviceType D, typename T>
void WinogradConvolution(const index_t batch,
                         const index_t height,
                         const index_t width,
                         const index_t in_channels,
                         const index_t out_channels,
                         const Padding padding,
                         const int block_size) {
  // srand(time(NULL));

  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, in_channels});
  net.AddRandomInput<D, float>("Filter", {out_channels, in_channels, 3, 3});
  net.AddRandomInput<D, T>("Bias", {out_channels});

  BufferToImage<D, T>(&net, "Input", "InputImage",
                      ops::BufferType::IN_OUT_CHANNEL);
  BufferToImage<D, T>(&net, "Filter", "FilterImage",
                      ops::BufferType::CONV2D_FILTER);
  BufferToImage<D, T>(&net, "Bias", "BiasImage", ops::BufferType::ARGUMENT);
  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("InputImage")
      .Input("FilterImage")
      .Input("BiasImage")
      .Output("OutputImage")
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  net.RunOp(D);

  // Transfer output
  ImageToBuffer<D, float>(&net, "OutputImage", "ConvOutput",
                          ops::BufferType::IN_OUT_CHANNEL);

  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("ConvOutput"));
  auto output_shape = expected->shape();

  // Winograd convolution
  // transform filter
  BufferToImage<D, T>(&net, "Filter", "WinoFilter",
                      ops::BufferType::WINOGRAD_FILTER, block_size);
  // transform input
  OpDefBuilder("WinogradTransform", "WinogradTransformTest")
      .Input("InputImage")
      .Output("WinoInput")
      .AddIntArg("padding", padding)
      .AddIntArg("wino_block_size", block_size)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(D);

  OpDefBuilder("InferConv2dShape", "InferConv2dShapeTest")
      .Input("InputImage")
      .Output("ShapeOutput")
      .AddIntArg("data_format", 0)
      .AddIntsArg("strides", {1, 1})
      .AddIntsArg("kernels", {static_cast<int>(out_channels),
                              static_cast<int>(in_channels),
                              3, 3})
      .AddIntArg("padding", padding)
      .OutputType({DataTypeToEnum<int32_t>::v()})
      .Finalize(net.NewOperatorDef());
  net.RunOp(D);

  // MatMul
  OpDefBuilder("MatMul", "MatMulTest")
      .Input("WinoFilter")
      .Input("WinoInput")
      .Output("WinoGemm")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  // Run on opencl
  net.RunOp(D);

  // Inverse transform
  OpDefBuilder("WinogradInverseTransform", "WinogradInverseTransformTest")
      .Input("WinoGemm")
      .Input("ShapeOutput")
      .Input("BiasImage")
      .AddIntArg("wino_block_size", block_size)
      .Output("WinoOutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(D);
  net.Sync();

  ImageToBuffer<D, float>(&net, "WinoOutputImage", "WinoOutput",
                          ops::BufferType::IN_OUT_CHANNEL);
  if (DataTypeToEnum<T>::value == DataType::DT_HALF) {
    ExpectTensorNear<float>(*expected, *net.GetOutput("WinoOutput"),
                            1e-2, 1e-2);
  } else {
    ExpectTensorNear<float>(*expected, *net.GetOutput("WinoOutput"),
                            1e-5, 1e-4);
  }
}
}  // namespace

TEST_F(WinogradConvolutionTest, AlignedConvolutionM2) {
  WinogradConvolution<DeviceType::GPU, float>(1, 32, 32, 3, 3,
                                                 Padding::VALID, 2);
  WinogradConvolution<DeviceType::GPU, float>(1, 32, 32, 3, 3,
                                                 Padding::SAME, 2);
}

TEST_F(WinogradConvolutionTest, UnAlignedConvolutionM2) {
  WinogradConvolution<DeviceType::GPU, float>(1, 61, 67, 31, 37,
                                                 Padding::VALID, 2);
  WinogradConvolution<DeviceType::GPU, float>(1, 61, 67, 37, 31,
                                                 Padding::SAME, 2);
}

TEST_F(WinogradConvolutionTest, BatchConvolutionM2) {
  WinogradConvolution<DeviceType::GPU, float>(3, 64, 64, 32, 32,
                                                 Padding::VALID, 2);
  WinogradConvolution<DeviceType::GPU, float>(5, 61, 67, 37, 31,
                                                 Padding::SAME, 2);
}

TEST_F(WinogradConvolutionTest, AlignedConvolutionM4) {
  WinogradConvolution<DeviceType::GPU, float>(1, 32, 32, 3, 3,
                                              Padding::VALID, 4);
  WinogradConvolution<DeviceType::GPU, float>(1, 32, 32, 3, 3,
                                              Padding::SAME, 4);
}

TEST_F(WinogradConvolutionTest, UnAlignedConvolutionM4) {
  WinogradConvolution<DeviceType::GPU, float>(1, 61, 67, 31, 37,
                                              Padding::VALID, 4);
  WinogradConvolution<DeviceType::GPU, float>(1, 61, 67, 37, 31,
                                              Padding::SAME, 4);
}

TEST_F(WinogradConvolutionTest, BatchConvolutionM4) {
  WinogradConvolution<DeviceType::GPU, float>(3, 107, 113, 5, 7,
                                              Padding::VALID, 4);
  WinogradConvolution<DeviceType::GPU, float>(5, 107, 113, 5, 7,
                                              Padding::SAME, 4);
}

namespace {
template <DeviceType D, typename T>
void WinogradConvolutionWithPad(const index_t batch,
                                const index_t height,
                                const index_t width,
                                const index_t in_channels,
                                const index_t out_channels,
                                const int padding,
                                const int block_size) {
  // srand(time(NULL));

  // Construct graph
  OpsTestNet net;
  // Add input data
  net.AddRandomInput<D, float>("Input", {batch, height, width, in_channels});
  net.AddRandomInput<D, float>("Filter", {out_channels, in_channels, 3, 3});
  net.AddRandomInput<D, float>("Bias", {out_channels});

  BufferToImage<D, T>(&net, "Input", "InputImage",
                      ops::BufferType::IN_OUT_CHANNEL);
  BufferToImage<D, T>(&net, "Filter", "FilterImage",
                      ops::BufferType::CONV2D_FILTER);
  BufferToImage<D, T>(&net, "Bias", "BiasImage", ops::BufferType::ARGUMENT);
  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("InputImage")
      .Input("FilterImage")
      .Input("BiasImage")
      .Output("OutputImage")
      .AddIntsArg("strides", {1, 1})
      .AddIntsArg("padding_values", {padding, padding})
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  net.RunOp(D);

  // Transfer output
  ImageToBuffer<D, float>(&net, "OutputImage", "ConvOutput",
                          ops::BufferType::IN_OUT_CHANNEL);
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("ConvOutput"));
  auto output_shape = expected->shape();

  // Winograd convolution
  // transform filter
  BufferToImage<D, T>(&net, "Filter", "WinoFilter",
                      ops::BufferType::WINOGRAD_FILTER, block_size);
  // transform input
  OpDefBuilder("WinogradTransform", "WinogradTransformTest")
      .Input("InputImage")
      .Output("WinoInput")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .AddIntsArg("padding_values", {padding, padding})
      .AddIntArg("wino_block_size", block_size)
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(D);

  OpDefBuilder("InferConv2dShape", "InferConv2dShapeTest")
      .Input("InputImage")
      .Output("ShapeOutput")
      .AddIntArg("data_format", 0)
      .AddIntsArg("strides", {1, 1})
      .AddIntsArg("kernels", {static_cast<int>(out_channels),
                              static_cast<int>(in_channels),
                              3, 3})
      .AddIntsArg("padding_values", {padding, padding})
      .OutputType({DataTypeToEnum<int32_t>::v()})
      .Finalize(net.NewOperatorDef());
  net.RunOp(D);

  // MatMul
  OpDefBuilder("MatMul", "MatMulTest")
      .Input("WinoFilter")
      .Input("WinoInput")
      .Output("WinoGemm")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  // Run on opencl
  net.RunOp(D);

  // Inverse transform
  OpDefBuilder("WinogradInverseTransform", "WinogradInverseTransformTest")
      .Input("WinoGemm")
      .Input("ShapeOutput")
      .Input("BiasImage")
      .AddIntArg("wino_block_size", block_size)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Output("WinoOutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(D);
  net.Sync();

  ImageToBuffer<D, float>(&net, "WinoOutputImage", "WinoOutput",
                          ops::BufferType::IN_OUT_CHANNEL);
  if (DataTypeToEnum<T>::value == DataType::DT_HALF) {
    ExpectTensorNear<float>(*expected, *net.GetOutput("WinoOutput"),
                            1e-2, 1e-2);
  } else {
    ExpectTensorNear<float>(*expected, *net.GetOutput("WinoOutput"),
                            1e-5, 1e-4);
  }
}
}  // namespace

TEST_F(WinogradConvolutionTest, AlignedConvolutionM2WithPad) {
  WinogradConvolutionWithPad<DeviceType::GPU, float>(1, 32, 32, 32, 16,
                                                     1, 2);
  WinogradConvolutionWithPad<DeviceType::GPU, float>(1, 32, 32, 32, 16,
                                                    2, 2);
}

TEST_F(WinogradConvolutionTest, UnAlignedConvolutionM2WithPad) {
  WinogradConvolutionWithPad<DeviceType::GPU, float>(1, 61, 67, 31, 37,
                                                     1, 2);
  WinogradConvolutionWithPad<DeviceType::GPU, float>(1, 61, 67, 37, 31,
                                                    2, 2);
}

TEST_F(WinogradConvolutionTest, BatchConvolutionWithM2Pad) {
  WinogradConvolutionWithPad<DeviceType::GPU, float>(3, 64, 64, 32, 32,
                                                     1, 2);
  WinogradConvolutionWithPad<DeviceType::GPU, float>(5, 61, 67, 37, 31,
                                                    2, 2);
}

TEST_F(WinogradConvolutionTest, AlignedConvolutionM4WithPad) {
  WinogradConvolutionWithPad<DeviceType::GPU, float>(1, 32, 32, 32, 16,
                                                     1, 4);
  WinogradConvolutionWithPad<DeviceType::GPU, float>(1, 32, 32, 32, 16,
                                                    2, 4);
}

TEST_F(WinogradConvolutionTest, UnAlignedConvolutionM4WithPad) {
  WinogradConvolutionWithPad<DeviceType::GPU, float>(1, 61, 67, 31, 37,
                                                     1, 4);
  WinogradConvolutionWithPad<DeviceType::GPU, float>(1, 61, 67, 37, 31,
                                                    2, 4);
}

TEST_F(WinogradConvolutionTest, BatchConvolutionWithM4Pad) {
  WinogradConvolutionWithPad<DeviceType::GPU, float>(3, 64, 64, 32, 32,
                                                     1, 4);
  WinogradConvolutionWithPad<DeviceType::GPU, float>(5, 61, 67, 37, 31,
                                                    2, 4);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
