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

#include <fstream>
#include <vector>

#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class DepthwiseDeconv2dOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void RunTestSimple(const int group,
                   const std::vector<index_t> &input_shape,
                   const std::vector<float> &input_data,
                   const std::vector<float> &bias_data,
                   const int stride,
                   const std::vector<int> &paddings,
                   const std::vector<index_t> &filter_shape,
                   const std::vector<float> &filter_data,
                   const std::vector<index_t> &expected_shape,
                   const std::vector<float> &expected_data) {
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);
  net.AddInputFromArray<D, float>("Filter", filter_shape, filter_data, true);
  net.TransformFilterDataFormat<D, float>(
      "Filter", DataFormat::HWOI, "FilterOIHW", DataFormat::OIHW);
  const index_t out_channels = expected_shape[3];
  net.AddInputFromArray<D, float>("Bias", {out_channels}, bias_data, true);

  if (D == DeviceType::GPU) {
    OpDefBuilder("DepthwiseDeconv2d", "DepthwiseDeconv2dTest")
        .Input("Input")
        .Input("FilterOIHW")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {stride, stride})
        .AddIntArg("group", group)
        .AddIntsArg("padding_values", paddings)
        .Finalize(net.NewOperatorDef());

    net.RunOp(D);
  } else {
    net.TransformDataFormat<DeviceType::CPU, float>(
        "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
    OpDefBuilder("DepthwiseDeconv2d", "DepthwiseDeconv2dTest")
        .Input("InputNCHW")
        .Input("FilterOIHW")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntArg("group", group)
        .AddIntsArg("strides", {stride, stride})
        .AddIntsArg("padding_values", paddings)
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>(
        "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);
  }

  auto expected = net.CreateTensor<float>(expected_shape, expected_data);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.0001);
}

template <DeviceType D>
void TestNHWCSimple3x3_DW() {
  RunTestSimple<D>(3,
                   {1, 3, 3, 3},
                   {1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1},
                   {0, 0, 0},
                   1, {0, 0},
                   {3, 3, 1, 3},
                   {1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1},
                   {1, 5, 5, 3},
                   {1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1,
                    2, 2, 2, 4, 4, 4, 6, 6, 6, 4, 4, 4, 2, 2, 2,
                    3, 3, 3, 6, 6, 6, 9, 9, 9, 6, 6, 6, 3, 3, 3,
                    2, 2, 2, 4, 4, 4, 6, 6, 6, 4, 4, 4, 2, 2, 2,
                    1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1});
}

template <DeviceType D>
void TestNHWCSimple3x3_Group() {
  RunTestSimple<D>(2,
                   {1, 3, 3, 4},
                   {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                    1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                    1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4},
                   {0, 0, 0, 0, 0, 0},
                   1, {0, 0},
                   {3, 3, 3, 4},
                   {1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
                    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
                    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
                    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
                    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
                    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
                    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
                    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1,
                    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1},
                   {1, 5, 5, 6},
                   {3, 6, 3, 7, 14, 7,
                    6, 12, 6, 14, 28, 14,
                    9, 18, 9, 21, 42, 21,
                    6, 12, 6, 14, 28, 14,
                    3, 6, 3, 7, 14, 7,
                    6, 12, 6, 14, 28, 14,
                    12, 24, 12, 28, 56, 28,
                    18, 36, 18, 42, 84, 42,
                    12, 24, 12, 28, 56, 28,
                    6, 12, 6, 14, 28, 14,
                    9, 18, 9, 21, 42, 21,
                    18, 36, 18, 42, 84, 42,
                    27, 54, 27, 63, 126, 63,
                    18, 36, 18, 42, 84, 42,
                    9, 18, 9, 21, 42, 21,
                    6, 12, 6, 14, 28, 14,
                    12, 24, 12, 28, 56, 28,
                    18, 36, 18, 42, 84, 42,
                    12, 24, 12, 28, 56, 28,
                    6, 12, 6, 14, 28, 14,
                    3, 6, 3, 7, 14, 7,
                    6, 12, 6, 14, 28, 14,
                    9, 18, 9, 21, 42, 21,
                    6, 12, 6, 14, 28, 14,
                    3, 6, 3, 7, 14, 7});
}
}  // namespace

TEST_F(DepthwiseDeconv2dOpTest, CPUSimple3X3Depthwise) {
  TestNHWCSimple3x3_DW<DeviceType::CPU>();
}

TEST_F(DepthwiseDeconv2dOpTest, CPUSimple3X3Group) {
  TestNHWCSimple3x3_Group<DeviceType::CPU>();
}

TEST_F(DepthwiseDeconv2dOpTest, GPUSimple3X3Depthwise) {
  TestNHWCSimple3x3_DW<DeviceType::GPU>();
}

namespace {
template <typename T>
void RandomTest(index_t batch,
                index_t channel,
                index_t height,
                index_t width,
                index_t kernel,
                int stride,
                int padding) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;
  int multiplier = 1;

  // Add input data
  std::vector<float> input_data(batch * height * width * channel);
  GenerateRandomRealTypeData({batch, height, width, channel}, &input_data);
  net.AddInputFromArray<DeviceType::GPU, float>("Input",
                                                {batch,
                                                 height,
                                                 width,
                                                 channel},
                                                input_data);
  std::vector<float> filter_data(kernel * kernel * channel * multiplier);
  GenerateRandomRealTypeData({multiplier, channel, kernel, kernel},
                             &filter_data);
  net.AddInputFromArray<DeviceType::GPU, float>(
      "Filter", {multiplier, channel, kernel, kernel}, filter_data, true,
      false);
  std::vector<float> bias_data(channel * multiplier);
  GenerateRandomRealTypeData({channel * multiplier}, &bias_data);
  net.AddInputFromArray<DeviceType::GPU, float>("Bias",
                                                {channel * multiplier},
                                                bias_data, true, false);

  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", DataFormat::NHWC, "InputNCHW", DataFormat::NCHW);
  OpDefBuilder("DepthwiseDeconv2d", "DepthwiseDeconv2dTest")
      .Input("InputNCHW")
      .Input("Filter")
      .Input("Bias")
      .Output("OutputNCHW")
      .AddIntsArg("strides", {stride, stride})
      .AddIntsArg("padding_values", {padding, padding})
      .AddIntArg("group", channel)
      .AddIntsArg("dilations", {1, 1})
      .AddStringArg("activation", "LEAKYRELU")
      .AddFloatArg("leakyrelu_coefficient", 0.1f)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<float>::value))
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(DeviceType::CPU);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", DataFormat::NCHW, "Output", DataFormat::NHWC);


  // Check
  auto expected = net.CreateTensor<float>();
  expected->Copy(*net.GetOutput("Output"));


  OpDefBuilder("DepthwiseDeconv2d", "DepthwiseDeconv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {stride, stride})
      .AddIntsArg("padding_values", {padding, padding})
      .AddIntArg("group", channel)
      .AddStringArg("activation", "LEAKYRELU")
      .AddFloatArg("leakyrelu_coefficient", 0.1f)
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  net.RunOp(DeviceType::GPU);

  if (DataTypeToEnum<T>::value == DT_FLOAT) {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
  } else {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-2);
  }
}

TEST_F(DepthwiseDeconv2dOpTest, RandomTestFloat) {
  RandomTest<float>(1, 32, 256, 256, 5, 1, 2);
  RandomTest<float>(1, 3, 256, 256, 5, 1, 1);
  RandomTest<float>(1, 3, 256, 256, 5, 2, 2);
  RandomTest<float>(1, 3, 256, 256, 5, 1, 3);
  RandomTest<float>(1, 3, 256, 256, 5, 2, 4);
  RandomTest<float>(1, 4, 256, 256, 5, 1, 1);
  RandomTest<float>(1, 4, 256, 256, 5, 2, 2);
  RandomTest<float>(1, 4, 256, 256, 5, 1, 3);
  RandomTest<float>(1, 4, 256, 256, 5, 2, 4);
}

TEST_F(DepthwiseDeconv2dOpTest, RandomTestHalf) {
  RandomTest<half>(1, 32, 256, 256, 5, 1, 2);
  RandomTest<half>(1, 3, 256, 256, 5, 1, 1);
  RandomTest<half>(1, 3, 256, 256, 5, 2, 2);
  RandomTest<half>(1, 3, 256, 256, 5, 1, 3);
  RandomTest<half>(1, 3, 256, 256, 5, 2, 4);
  RandomTest<half>(1, 4, 256, 256, 5, 1, 1);
  RandomTest<half>(1, 4, 256, 256, 5, 2, 2);
  RandomTest<half>(1, 4, 256, 256, 5, 1, 3);
  RandomTest<half>(1, 4, 256, 256, 5, 2, 4);
}

}  // namespace
}  // namespace test
}  // namespace ops
}  // namespace mace
