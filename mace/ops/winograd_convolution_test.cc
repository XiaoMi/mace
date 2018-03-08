//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <fstream>
#include "mace/core/operator.h"
#include "mace/kernels/conv_pool_2d_util.h"
#include "mace/ops/ops_test_util.h"

namespace mace {

class WinogradConvlutionTest : public OpsTestBase {};

void TransposeFilter(const std::vector<float> &input,
                     const std::vector<index_t> &input_shape,
                     std::vector<float> &output) {
  output.resize(input.size());

  const float *input_ptr = input.data();
  for (index_t h = 0; h < input_shape[0]; ++h) {
    for (index_t w = 0; w < input_shape[1]; ++w) {
      for (index_t oc = 0; oc < input_shape[2]; ++oc) {
        for (index_t ic = 0; ic < input_shape[3]; ++ic) {
          int offset = ((oc * input_shape[3] + ic) * input_shape[0] + h) *
                           input_shape[1] +
                       w;
          output[offset] = *input_ptr;
          ++input_ptr;
        }
      }
    }
  }
}

template <DeviceType D, typename T>
void WinogradConvolution(const index_t batch,
                         const index_t height,
                         const index_t width,
                         const index_t in_channels,
                         const index_t out_channels,
                         const Padding padding) {
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;
  // Add input data
  std::vector<float> filter_data;
  std::vector<index_t> filter_shape = {3, 3, out_channels, in_channels};
  GenerateRandomRealTypeData<float>(filter_shape, filter_data);
  net.AddRandomInput<D, float>("Input", {batch, height, width, in_channels});
  net.AddInputFromArray<D, float>("Filter", filter_shape, filter_data);
  net.AddRandomInput<D, T>("Bias", {out_channels});

  BufferToImage<D, T>(net, "Input", "InputImage",
                      kernels::BufferType::IN_OUT_CHANNEL);
  BufferToImage<D, T>(net, "Filter", "FilterImage",
                      kernels::BufferType::CONV2D_FILTER);
  BufferToImage<D, T>(net, "Bias", "BiasImage", kernels::BufferType::ARGUMENT);
  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("InputImage")
      .Input("FilterImage")
      .Input("BiasImage")
      .Output("OutputImage")
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  net.RunOp(D);

  // Transfer output
  ImageToBuffer<D, T>(net, "OutputImage", "ConvOutput",
                      kernels::BufferType::IN_OUT_CHANNEL);
  Tensor expected;
  expected.Copy(*net.GetOutput("ConvOutput"));
  auto output_shape = expected.shape();

  // Winograd convolution
  // transform filter
  std::vector<float> wino_filter_data;
  TransposeFilter(filter_data, filter_shape, wino_filter_data);
  net.AddInputFromArray<D, float>(
      "WinoFilterData", {out_channels, in_channels, 3, 3}, wino_filter_data);
  BufferToImage<D, T>(net, "WinoFilterData", "WinoFilter",
                      kernels::BufferType::WINOGRAD_FILTER);

  // transform input
  OpDefBuilder("WinogradTransform", "WinogradTransformTest")
      .Input("InputImage")
      .Output("WinoInput")
      .AddIntArg("padding", padding)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Run on opencl
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
      .Input("BiasImage")
      .AddIntArg("batch", batch)
      .AddIntArg("height", output_shape[1])
      .AddIntArg("width", output_shape[2])
      .Output("WinoOutputImage")
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(D);
  net.Sync();

  ImageToBuffer<D, float>(net, "WinoOutputImage", "WinoOutput",
                          kernels::BufferType::IN_OUT_CHANNEL);
  if (DataTypeToEnum<T>::value == DataType::DT_HALF) {
    ExpectTensorNear<float>(expected, *net.GetOutput("WinoOutput"), 1e-1);
  } else {
    ExpectTensorNear<float>(expected, *net.GetOutput("WinoOutput"), 1e-4);
  }
}

TEST_F(WinogradConvlutionTest, AlignedConvolution) {
  WinogradConvolution<DeviceType::OPENCL, float>(1, 32, 32, 32, 16,
                                                 Padding::VALID);
  WinogradConvolution<DeviceType::OPENCL, float>(1, 32, 32, 32, 16,
                                                 Padding::SAME);
}

TEST_F(WinogradConvlutionTest, UnAlignedConvolution) {
  WinogradConvolution<DeviceType::OPENCL, float>(1, 61, 67, 31, 37,
                                                 Padding::VALID);
  WinogradConvolution<DeviceType::OPENCL, float>(1, 61, 67, 37, 31,
                                                 Padding::SAME);
}

TEST_F(WinogradConvlutionTest, BatchConvolution) {
  WinogradConvolution<DeviceType::OPENCL, float>(3, 64, 64, 32, 32,
                                                 Padding::VALID);
  WinogradConvolution<DeviceType::OPENCL, float>(5, 61, 67, 37, 31,
                                                 Padding::SAME);
}
}
