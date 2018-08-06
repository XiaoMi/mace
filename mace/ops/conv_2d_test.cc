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
#include <vector>

#include "mace/kernels/quantize.h"
#include "mace/ops/conv_2d.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class Conv2dOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void TestNHWCSimple3x3VALID() {
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, T>(
      "Input", {1, 3, 3, 2},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, T>(
      "Filter", {1, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<D, T>("Bias", {1}, {0.1f});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);
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
        .Output("OutputImage")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());

    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, T>(&net, "OutputImage", "Output",
                        kernels::BufferType::IN_OUT_CHANNEL);

  } else {
    MACE_NOT_IMPLEMENTED;
  }

  auto expected = CreateTensor<float>({1, 1, 1, 1}, {18.1f});
  ExpectTensorNear<float, T>(*expected, *net.GetOutput("Output"), 1e-5);
}

template <DeviceType D, typename T>
void TestNHWCSimple3x3SAME() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>(
      "Input", {1, 3, 3, 2},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, T>(
      "Filter", {1, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<D, T>("Bias", {1}, {0.1f});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);
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
        .Output("OutputImage")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, T>(&net, "OutputImage", "Output",
                        kernels::BufferType::IN_OUT_CHANNEL);

  } else {
    MACE_NOT_IMPLEMENTED;
  }

  auto expected = CreateTensor<float>(
      {1, 3, 3, 1},
      {8.1f, 12.1f, 8.1f, 12.1f, 18.1f, 12.1f, 8.1f, 12.1f, 8.1f});

  ExpectTensorNear<float, T>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(Conv2dOpTest, CPUSimple) {
  TestNHWCSimple3x3VALID<DeviceType::CPU, float>();
  TestNHWCSimple3x3SAME<DeviceType::CPU, float>();
}

TEST_F(Conv2dOpTest, OPENCLSimple) {
  TestNHWCSimple3x3VALID<DeviceType::GPU, float>();
  TestNHWCSimple3x3SAME<DeviceType::GPU, float>();
}

namespace {
template <DeviceType D, typename T>
void TestNHWCSimple3x3WithoutBias() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>(
      "Input", {1, 3, 3, 2},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, T>(
      "Filter", {1, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Filter", "FilterImage",
                        kernels::BufferType::CONV2D_FILTER);

    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    // Transfer output
    ImageToBuffer<D, T>(&net, "OutputImage", "Output",
                        kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 1}, {18.0f});

  ExpectTensorNear<float, T>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(Conv2dOpTest, CPUWithoutBias) {
  TestNHWCSimple3x3WithoutBias<DeviceType::CPU, float>();
}

TEST_F(Conv2dOpTest, OPENCLWithoutBias) {
  TestNHWCSimple3x3WithoutBias<DeviceType::GPU, float>();
}

namespace {
template <DeviceType D, typename T>
void TestNHWCCombined3x3() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>(
      "Input", {1, 5, 5, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, T>(
      "Filter", {2, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
       0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
  net.AddInputFromArray<D, T>("Bias", {2}, {0.1f, 0.2f});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("Conv2D", "Conv2DTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Filter", "FilterImage",
                        kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, T>(&net, "Bias", "BiasImage",
                        kernels::BufferType::ARGUMENT);

    OpDefBuilder("Conv2D", "Conv2DTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    ImageToBuffer<D, T>(&net, "OutputImage", "Output",
                        kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = CreateTensor<float>(
      {1, 3, 3, 2}, {8.1f, 4.2f, 12.1f, 6.2f, 8.1f, 4.2f, 12.1f, 6.2f, 18.1f,
                     9.2f, 12.1f, 6.2f, 8.1f, 4.2f, 12.1f, 6.2f, 8.1f, 4.2f});
  ExpectTensorNear<float, T>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(Conv2dOpTest, CPUStride2) {
  TestNHWCCombined3x3<DeviceType::CPU, float>();
}

TEST_F(Conv2dOpTest, OPENCLStride2) {
  TestNHWCCombined3x3<DeviceType::GPU, float>();
}

namespace {
template <DeviceType D, typename T>
void TestFusedNHWCSimple3x3VALID() {
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 3, 3, 2},
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});
  net.AddInputFromArray<D, float>(
      "Filter", {1, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<D, float>("Bias", {1}, {-0.1f});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Filter", "FilterImage",
                        kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, T>(&net, "Bias", "BiasImage",
                        kernels::BufferType::ARGUMENT);
    OpDefBuilder("Conv2D", "Conv2DTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());

    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);

  } else {
    MACE_NOT_IMPLEMENTED;
  }

  auto expected = CreateTensor<float>({1, 1, 1, 1}, {0.0f});
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"));
}
template <DeviceType D, typename T>
void TestFusedNHWCSimple3x3WithoutBias() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 3, 3, 2},
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1});
  net.AddInputFromArray<D, float>(
      "Filter", {1, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("Conv2D", "Conv2DTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Filter", "FilterImage",
                        kernels::BufferType::CONV2D_FILTER);

    OpDefBuilder("Conv2D", "Conv2DTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .AddStringArg("activation", "RELU")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 1}, {0.0f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"));
}

}  // namespace

TEST_F(Conv2dOpTest, FusedCPUSimple) {
  TestFusedNHWCSimple3x3VALID<DeviceType::CPU, float>();
  TestFusedNHWCSimple3x3WithoutBias<DeviceType::CPU, float>();
}

TEST_F(Conv2dOpTest, FusedOPENCLSimple) {
  TestFusedNHWCSimple3x3VALID<DeviceType::GPU, float>();
  TestFusedNHWCSimple3x3WithoutBias<DeviceType::GPU, float>();
}

namespace {
template <DeviceType D>
void TestConv1x1() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 3, 10, 5},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, float>(
      "Filter", {2, 5, 1, 1},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});
  net.AddInputFromArray<D, float>("Bias", {2}, {0.1f, 0.2f});

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("Conv2D", "Conv2DTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);
  } else if (D == DeviceType::GPU) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, float>(&net, "Filter", "FilterImage",
                            kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, float>(&net, "Bias", "BiasImage",
                            kernels::BufferType::ARGUMENT);

    OpDefBuilder("Conv2D", "Conv2DTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = CreateTensor<float>(
      {1, 3, 10, 2},
      {5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
       5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
       5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
       5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
       5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f,
       5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f, 5.1f, 10.2f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}
}  // namespace

TEST_F(Conv2dOpTest, CPUConv1x1) { TestConv1x1<DeviceType::CPU>(); }

TEST_F(Conv2dOpTest, OPENCLConv1x1) { TestConv1x1<DeviceType::GPU>(); }

namespace {
template <DeviceType D, typename T>
void TestComplexConvNxNS12(const std::vector<index_t> &shape,
                           const int stride) {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type) {
    // generate random input
    static unsigned int seed = time(NULL);
    index_t batch = 3 + (rand_r(&seed) % 10);
    index_t height = shape[0];
    index_t width = shape[1];
    index_t input_channels = shape[2] + (rand_r(&seed) % 10);
    index_t output_channels = shape[3] + (rand_r(&seed) % 10);

    OpsTestNet net;

    // Add input data
    net.AddRandomInput<D, T>("Input", {batch, height, width, input_channels});
    net.AddRandomInput<D, T>(
        "Filter", {output_channels, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<D, T>("Bias", {output_channels});
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);

    // Construct graph
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());

    // run on cpu
    net.RunOp();

    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);

    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run on gpu
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
        .Output("OutputImage")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run on device
    net.RunOp(D);

    ImageToBuffer<D, T>(&net, "OutputImage", "OPENCLOutput",
                        kernels::BufferType::IN_OUT_CHANNEL);
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-4,
                            1e-4);
  };

  for (int kernel_size : {1, 3, 5, 7}) {
    func(kernel_size, kernel_size, stride, stride, VALID);
    func(kernel_size, kernel_size, stride, stride, SAME);
  }
}
}  // namespace

TEST_F(Conv2dOpTest, OPENCLAlignedConvNxNS12) {
  TestComplexConvNxNS12<DeviceType::GPU, float>({32, 16, 16, 32}, 1);
  TestComplexConvNxNS12<DeviceType::GPU, float>({32, 16, 16, 32}, 2);
}

TEST_F(Conv2dOpTest, OPENCLUnalignedConvNxNS12) {
  TestComplexConvNxNS12<DeviceType::GPU, float>({17, 113, 5, 7}, 1);
  TestComplexConvNxNS12<DeviceType::GPU, float>({17, 113, 5, 7}, 2);
}

TEST_F(Conv2dOpTest, OPENCLUnalignedConvNxNS34) {
  TestComplexConvNxNS12<DeviceType::GPU, float>({31, 113, 13, 17}, 3);
  TestComplexConvNxNS12<DeviceType::GPU, float>({32, 32, 13, 17}, 4);
}

namespace {
template <DeviceType D>
void TestHalfComplexConvNxNS12(const std::vector<index_t> &input_shape,
                               const std::vector<index_t> &filter_shape,
                               const std::vector<int> &dilations) {
  testing::internal::LogToStderr();
  srand(time(NULL));

  auto func = [&](int stride_h, int stride_w, Padding padding) {
    // generate random input
    index_t batch = 3;
    index_t height = input_shape[0];
    index_t width = input_shape[1];
    index_t kernel_h = filter_shape[0];
    index_t kernel_w = filter_shape[1];
    index_t input_channels = filter_shape[2];
    index_t output_channels = filter_shape[3];
    // Construct graph
    OpsTestNet net;

    std::vector<float> float_input_data;
    GenerateRandomRealTypeData({batch, height, width, input_channels},
                               &float_input_data);
    std::vector<float> float_filter_data;
    GenerateRandomRealTypeData(
        {kernel_h, kernel_w, output_channels, input_channels},
        &float_filter_data);
    std::vector<float> float_bias_data;
    GenerateRandomRealTypeData({output_channels}, &float_bias_data);
    // Add input data
    net.AddInputFromArray<D, float>(
        "Input", {batch, height, width, input_channels}, float_input_data);
    net.AddInputFromArray<D, float>(
        "Filter", {output_channels, input_channels, kernel_h, kernel_w},
        float_filter_data);
    net.AddInputFromArray<D, float>("Bias", {output_channels}, float_bias_data);

    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);

    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", padding)
        .AddIntsArg("dilations", {dilations[0], dilations[1]})
        .Finalize(net.NewOperatorDef());

    // run on cpu
    net.RunOp();

    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);

    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run on gpu
    BufferToImage<D, half>(&net, "Input", "InputImage",
                           kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, half>(&net, "Filter", "FilterImage",
                           kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, half>(&net, "Bias", "BiasImage",
                           kernels::BufferType::ARGUMENT);

    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", padding)
        .AddIntsArg("dilations", {dilations[0], dilations[1]})
        .AddIntArg("T", static_cast<int>(DataType::DT_HALF))
        .Finalize(net.NewOperatorDef());
    // Run on device
    net.RunOp(D);

    ImageToBuffer<D, float>(&net, "OutputImage", "OPENCLOutput",
                            kernels::BufferType::IN_OUT_CHANNEL);

    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-2,
                            1e-1);
  };

  func(1, 1, VALID);
  func(1, 1, SAME);
  if (dilations[0] == 1) {
    func(2, 2, VALID);
    func(2, 2, SAME);
  }
}
}  // namespace

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv1x1S12) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({32, 32}, {1, 1, 32, 64}, {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv3x3S12) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({32, 32}, {3, 3, 32, 64}, {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv5x5S12) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({32, 32}, {5, 5, 3, 64}, {1, 1});
  TestHalfComplexConvNxNS12<DeviceType::GPU>({32, 32}, {5, 5, 3, 63}, {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv1x7S1) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({17, 17}, {1, 7, 192, 192},
                                             {1, 1});
  TestHalfComplexConvNxNS12<DeviceType::GPU>({17, 17}, {1, 7, 192, 191},
                                             {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv7x1S1) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({17, 17}, {7, 1, 192, 192},
                                             {1, 1});
  TestHalfComplexConvNxNS12<DeviceType::GPU>({17, 17}, {7, 1, 160, 192},
                                             {1, 1});
  TestHalfComplexConvNxNS12<DeviceType::GPU>({17, 17}, {7, 1, 160, 191},
                                             {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv7x7S12) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({32, 32}, {7, 7, 3, 64}, {1, 1});
  TestHalfComplexConvNxNS12<DeviceType::GPU>({32, 32}, {7, 7, 3, 63}, {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv15x1S12) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({32, 32}, {15, 1, 256, 2}, {1, 1});
  TestHalfComplexConvNxNS12<DeviceType::GPU>({64, 64}, {15, 1, 64, 2}, {1, 1});
  TestHalfComplexConvNxNS12<DeviceType::GPU>({256, 256}, {15, 1, 32, 2},
                                             {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv1x15S12) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({32, 32}, {1, 15, 256, 2}, {1, 1});
  TestHalfComplexConvNxNS12<DeviceType::GPU>({256, 256}, {1, 15, 32, 2},
                                             {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfUnalignedConv1x1S12) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({107, 113}, {1, 1, 5, 7}, {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfUnalignedConv3x3S12) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({107, 113}, {3, 3, 5, 7}, {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfConv5x5Dilation2) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({64, 64}, {5, 5, 16, 16}, {2, 2});
}

TEST_F(Conv2dOpTest, OPENCLHalfConv7x7Dilation2) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({64, 64}, {7, 7, 16, 16}, {2, 2});
}

TEST_F(Conv2dOpTest, OPENCLHalfConv7x7Dilation4) {
  TestHalfComplexConvNxNS12<DeviceType::GPU>({63, 67}, {7, 7, 16, 16}, {4, 4});
}

namespace {
template <DeviceType D, typename T>
void TestDilationConvNxN(const std::vector<index_t> &shape,
                         const int dilation_rate) {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type) {
    srand(time(NULL));

    // generate random input
    index_t batch = 1;
    index_t height = shape[0];
    index_t width = shape[1];
    index_t input_channels = shape[2];
    index_t output_channels = shape[3];

    OpsTestNet net;

    // Add input data
    net.AddRandomInput<D, T>("Input", {batch, height, width, input_channels});
    net.AddRandomInput<D, T>(
        "Filter", {output_channels, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<D, T>("Bias", {output_channels});

    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);

    // Construct graph
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", {dilation_rate, dilation_rate})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());

    // run on cpu
    net.RunOp();
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);

    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run on gpu
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
        .Output("OutputImage")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", {dilation_rate, dilation_rate})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run on device
    net.RunOp(D);

    ImageToBuffer<D, T>(&net, "OutputImage", "OPENCLOutput",
                        kernels::BufferType::IN_OUT_CHANNEL);
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-4,
                            1e-4);
  };

  for (int kernel_size : {3}) {
    for (int stride : {1}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }
}
}  // namespace

TEST_F(Conv2dOpTest, OPENCLAlignedDilation2) {
  TestDilationConvNxN<DeviceType::GPU, float>({32, 32, 32, 64}, 2);
}

TEST_F(Conv2dOpTest, OPENCLAligned2Dilation4) {
  TestDilationConvNxN<DeviceType::GPU, float>({128, 128, 16, 16}, 4);
}

TEST_F(Conv2dOpTest, OPENCLUnalignedDilation4) {
  TestDilationConvNxN<DeviceType::GPU, float>({107, 113, 5, 7}, 4);
}

namespace {
template <DeviceType D>
void TestGeneralHalfAtrousConv(const std::vector<index_t> &image_shape,
                               const std::vector<index_t> &filter_shape,
                               const std::vector<int> &dilations) {
  testing::internal::LogToStderr();
  auto func = [&](int stride_h, int stride_w, Padding type) {
    srand(time(NULL));

    // generate random input
    index_t batch = 1;
    index_t height = image_shape[0];
    index_t width = image_shape[1];
    index_t kernel_h = filter_shape[0];
    index_t kernel_w = filter_shape[1];
    index_t output_channels = filter_shape[2];
    index_t input_channels = filter_shape[3];

    OpsTestNet net;

    // Add input data
    net.AddRandomInput<D, float>("Input",
                                 {batch, height, width, input_channels});
    net.AddRandomInput<D, float>(
        "Filter", {output_channels, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<D, float>("Bias", {output_channels});

    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    // Construct graph
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", dilations)
        .Finalize(net.NewOperatorDef());

    // run on cpu
    net.RunOp();

    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);
    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run on gpu
    BufferToImage<D, half>(&net, "Input", "InputImage",
                           kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, half>(&net, "Filter", "FilterImage",
                           kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, half>(&net, "Bias", "BiasImage",
                           kernels::BufferType::ARGUMENT);

    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", dilations)
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<half>::value))
        .Finalize(net.NewOperatorDef());
    // Run on device
    net.RunOp(D);

    ImageToBuffer<D, float>(&net, "OutputImage", "OPENCLOutput",
                            kernels::BufferType::IN_OUT_CHANNEL);
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-2,
                            1e-1);
  };

  func(1, 1, VALID);
  func(1, 1, SAME);
}
}  // namespace

TEST_F(Conv2dOpTest, OPENCLHalf7X7AtrousConvD2) {
  TestGeneralHalfAtrousConv<DeviceType::GPU>({32, 32}, {7, 7, 16, 3}, {2, 2});
}

TEST_F(Conv2dOpTest, OPENCLHalf15X15AtrousConvD4) {
  TestGeneralHalfAtrousConv<DeviceType::GPU>({63, 71}, {15, 15, 16, 16},
                                             {2, 2});
}

namespace {
template <DeviceType D, typename T>
void TestArbitraryPadConvNxN(const std::vector<index_t> &shape,
                             const std::vector<int> &paddings) {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w) {
    srand(time(NULL));

    // generate random input
    index_t batch = 1;
    index_t height = shape[0];
    index_t width = shape[1];
    index_t input_channels = shape[2];
    index_t output_channels = shape[3];

    OpsTestNet net;

    // Add input data
    net.AddRandomInput<D, T>("Input", {batch, height, width, input_channels});
    net.AddRandomInput<D, T>(
        "Filter", {output_channels, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<D, T>("Bias", {output_channels});

    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    // Construct graph
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntsArg("padding_values", paddings)
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());

    // run on cpu
    net.RunOp();

    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);

    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run on gpu
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
        .Output("OutputImage")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntsArg("padding_values", paddings)
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run on device
    net.RunOp(D);

    ImageToBuffer<D, T>(&net, "OutputImage", "OPENCLOutput",
                        kernels::BufferType::IN_OUT_CHANNEL);
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-4,
                            1e-4);
  };

  for (int kernel_size : {3, 5, 7}) {
    for (int stride : {2, 3}) {
      func(kernel_size, kernel_size, stride, stride);
    }
  }
}
}  // namespace

TEST_F(Conv2dOpTest, OPENCLAlignedPad1) {
  TestArbitraryPadConvNxN<DeviceType::GPU, float>({32, 32, 32, 64}, {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLAlignedPad2) {
  TestArbitraryPadConvNxN<DeviceType::GPU, float>({128, 128, 16, 16}, {2, 2});
}

TEST_F(Conv2dOpTest, OPENCLUnalignedPad4) {
  TestArbitraryPadConvNxN<DeviceType::GPU, float>({107, 113, 5, 7}, {4, 4});
}

namespace {

void TestQuantSimple3x3() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, uint8_t>(
      "Filter", {1, 3, 3, 2},
      {102, 150, 123, 135, 1, 216, 137, 47, 53, 75, 145, 130, 171, 62, 255,
       122, 72, 211}, 0.0226, 127);
  net.AddInputFromArray<DeviceType::CPU, uint8_t>(
      "Input", {1, 3, 3, 2},
      {1, 75, 117, 161, 127, 119, 94, 151, 203, 151, 84, 61, 55, 142, 113, 139,
       3, 255}, 0.0204, 93);

  net.AddInputFromArray<DeviceType::CPU, int32_t>("Bias", {1}, {2});
  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DT_UINT8))
      .Finalize(net.NewOperatorDef());

  net.Setup(DeviceType::CPU);
  Tensor *output = net.GetTensor("Output");
  output->SetScale(0.000711);
  output->SetZeroPoint(1);
  // Run
  net.Run();
  // Check
  auto expected = CreateTensor<uint8_t>({1, 1, 1, 1}, {230});
  ExpectTensorNear<uint8_t>(*expected, *output);
}

void TestQuant(const index_t batch,
               const index_t out_channels,
               const index_t in_channels,
               const index_t in_height,
               const index_t in_width,
               const index_t k_height,
               const index_t k_width,
               enum Padding padding_type,
               const std::vector<int> &strides) {
  OpsTestNet net;
  net.AddRandomInput<CPU, float>("Input", {batch, in_height, in_width,
                                           in_channels});
  net.AddRandomInput<CPU, float>("Filter", {out_channels, k_height, k_width,
                                            in_channels});
  net.AddRandomInput<CPU, float>("Bias", {out_channels});
  net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                  NCHW);
  net.TransformDataFormat<DeviceType::CPU, float>("Filter", OHWI, "FilterOIHW",
                                                  OIHW);

  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("InputNCHW")
      .Input("FilterOIHW")
      .Input("Bias")
      .Output("OutputNCHW")
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding_type)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DT_FLOAT))
      .Finalize(net.NewOperatorDef());
  net.RunOp(CPU);
  net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                  "Output", NHWC);

  OpDefBuilder("Quantize", "QuantizeFilter")
      .Input("Filter")
      .Output("QuantizedFilter")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Quantize", "QuantizeInput")
      .Input("Input")
      .Output("QuantizedInput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  OpDefBuilder("Quantize", "QuantizeOutput")
      .Input("Output")
      .Output("ExpectedQuantizedOutput")
      .OutputType({DT_UINT8})
      .AddIntArg("T", DT_UINT8)
      .AddIntArg("non_zero", true)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  Tensor *q_filter = net.GetTensor("QuantizedFilter");
  Tensor *q_input = net.GetTensor("QuantizedInput");
  Tensor *bias = net.GetTensor("Bias");
  auto bias_data = bias->data<float>();
  std::vector<int32_t> q_bias(bias->size());
  kernels::QuantizeWithScaleAndZeropoint(
      bias_data, bias->size(), q_input->scale() * q_filter->scale(), 0,
      q_bias.data());
  net.AddInputFromArray<DeviceType::CPU, int32_t>("QuantizedBias",
                                                  {out_channels}, q_bias);
  OpDefBuilder("Conv2D", "QuantizeConv2dTest")
      .Input("QuantizedInput")
      .Input("QuantizedFilter")
      .Input("QuantizedBias")
      .Output("QuantizedOutput")
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding_type)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DT_UINT8))
      .Finalize(net.NewOperatorDef());
  net.Setup(DeviceType::CPU);
  Tensor *eq_output = net.GetTensor("ExpectedQuantizedOutput");
  Tensor *q_output = net.GetTensor("QuantizedOutput");
  q_output->SetScale(eq_output->scale());
  q_output->SetZeroPoint(eq_output->zero_point());
  net.Run();

  OpDefBuilder("Dequantize", "DeQuantizeTest")
      .Input("QuantizedOutput")
      .Output("DequantizedOutput")
      .OutputType({DT_FLOAT})
      .AddIntArg("T", DT_UINT8)
      .Finalize(net.NewOperatorDef());
  net.RunOp();

  // Check
  ExpectTensorSimilar<float>(*net.GetOutput("Output"),
                             *net.GetTensor("DequantizedOutput"), 0.01);
}
}  // namespace

TEST_F(Conv2dOpTest, Quant) {
  TestQuantSimple3x3();
  TestQuant(1, 128, 64, 32, 32, 1, 1, VALID, {1, 1});
  TestQuant(1, 128, 64, 32, 32, 3, 3, VALID, {1, 1});
  TestQuant(1, 128, 64, 32, 32, 3, 3, SAME, {1, 1});
  TestQuant(1, 128, 64, 32, 32, 3, 3, FULL, {1, 1});
  TestQuant(1, 128, 64, 32, 32, 3, 3, SAME, {2, 2});
  TestQuant(1, 129, 63, 33, 31, 3, 3, SAME, {1, 1});
  TestQuant(9, 128, 64, 32, 32, 3, 3, SAME, {1, 1});
  TestQuant(1, 128, 64, 32, 32, 1, 5, SAME, {1, 1});
  TestQuant(1, 128, 64, 32, 32, 5, 5, SAME, {1, 1});
  TestQuant(1, 128, 64, 32, 32, 5, 1, SAME, {1, 1});
  TestQuant(1, 128, 64, 32, 32, 7, 7, SAME, {1, 1});
  TestQuant(1, 128, 64, 32, 32, 7, 7, SAME, {2, 2});
  TestQuant(1, 128, 64, 32, 32, 7, 7, SAME, {3, 3});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
