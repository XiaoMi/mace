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

#include "mace/ops/deconv_2d.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class Deconv2dOpTest : public OpsTestBase {};

namespace {
template <DeviceType D>
void RunTestSimple(const std::vector<index_t> &input_shape,
                   const std::vector<float> &input_data,
                   const int stride,
                   Padding padding,
                   const std::vector<int> &padding_size,
                   const std::vector<int> &output_shape,
                   const std::vector<index_t> &filter_shape,
                   const std::vector<float> &filter_data,
                   const std::vector<index_t> &expected_shape,
                   const std::vector<float> &expected_data) {
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);
  net.AddInputFromArray<D, float>("Filter", filter_shape, filter_data);
  net.TransformDataFormat<D, float>("Filter", HWOI, "FilterOIHW", OIHW);
  bool from_caffe = output_shape.size() != 4;
  if (D == DeviceType::GPU) {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, float>(&net, "FilterOIHW", "FilterImage",
                            kernels::BufferType::CONV2D_FILTER);
    OpDefBuilder("Deconv2D", "Deconv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {stride, stride})
        .AddIntArg("padding", padding)
        .AddIntsArg("padding_values", padding_size)
        .AddIntsArg("output_shape", output_shape)
        .AddIntArg("from_caffe", from_caffe)
        .Finalize(net.NewOperatorDef());

    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);
  } else {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("Deconv2D", "Deconv2dTest")
        .Input("InputNCHW")
        .Input("FilterOIHW")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {stride, stride})
        .AddIntArg("padding", padding)
        .AddIntsArg("padding_values", padding_size)
        .AddIntsArg("output_shape", output_shape)
        .AddIntArg("from_caffe", from_caffe)
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);
  }

  auto expected = CreateTensor<float>(expected_shape, expected_data);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.0001);
}

template <DeviceType D>
void TestNHWCSimple3x3SAME_S1() {
  RunTestSimple<D>({1, 3, 3, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, 1, Padding::SAME,
                   {0, 0}, {1, 3, 3, 3}, {3, 3, 3, 1},
                   {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                   {1, 3, 3, 3}, {4, 4, 4, 6, 6, 6, 4, 4, 4, 6, 6, 6, 9, 9,
                                  9, 6, 6, 6, 4, 4, 4, 6, 6, 6, 4, 4, 4});
  RunTestSimple<D>({1, 3, 3, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, 1, Padding::VALID,
                   {2, 2}, {0}, {3, 3, 3, 1},
                   {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                   {1, 3, 3, 3}, {4, 4, 4, 6, 6, 6, 4, 4, 4, 6, 6, 6, 9, 9,
                                  9, 6, 6, 6, 4, 4, 4, 6, 6, 6, 4, 4, 4});
  RunTestSimple<D>({1, 3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, Padding::SAME,
                   {0, 0}, {1, 3, 3, 3}, {3, 3, 3, 1},
                   {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
                   {1, 3, 3, 3}, {54,  66,  78,  126, 147, 168, 130, 146, 162,
                                  198, 225, 252, 405, 450, 495, 366, 399, 432,
                                  354, 378, 402, 630, 669, 708, 502, 530, 558});
  RunTestSimple<D>(
      {1, 3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, Padding::SAME, {2, 2}, {0},
      {3, 3, 3, 1}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
      {1, 3, 3, 3},
      {54,  66,  78,  126, 147, 168, 130, 146, 162, 198, 225, 252, 405, 450,
       495, 366, 399, 432, 354, 378, 402, 630, 669, 708, 502, 530, 558});
}

template <DeviceType D>
void TestNHWCSimple3x3SAME_S2() {
  RunTestSimple<D>(
      {1, 3, 3, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, 2, Padding::SAME, {0, 0},
      {1, 6, 6, 3}, {3, 3, 3, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 6, 6, 3},
      {1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4,
       4, 2, 2, 2, 4, 4, 4, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1,
       2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 2, 2, 2, 4, 4, 4, 2,
       2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1});
  RunTestSimple<D>(
      {1, 3, 3, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, 2, Padding::SAME, {2, 2}, {0},
      {3, 3, 3, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 5, 5, 3}, {1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 4,
                     4, 4, 2, 2, 2, 4, 4, 4, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1,
                     1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 4, 4, 4, 2, 2, 2, 4, 4, 4,
                     2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1});
  RunTestSimple<D>(
      {1, 3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 2, Padding::SAME, {0, 0},
      {1, 6, 6, 3}, {3, 3, 3, 1},
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
       15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
      {1, 6, 6, 3},
      {1,   2,   3,   4,   5,   6,   9,   12,  15,  8,   10,  12,  17,  22,
       27,  12,  15,  18,  10,  11,  12,  13,  14,  15,  36,  39,  42,  26,
       28,  30,  62,  67,  72,  39,  42,  45,  23,  28,  33,  38,  43,  48,
       96,  108, 120, 64,  71,  78,  148, 164, 180, 90,  99,  108, 40,  44,
       48,  52,  56,  60,  114, 123, 132, 65,  70,  75,  140, 151, 162, 78,
       84,  90,  83,  94,  105, 116, 127, 138, 252, 276, 300, 142, 155, 168,
       304, 332, 360, 168, 183, 198, 70,  77,  84,  91,  98,  105, 192, 207,
       222, 104, 112, 120, 218, 235, 252, 117, 126, 135});
  RunTestSimple<D>(
      {1, 3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 2, Padding::SAME, {2, 2}, {0},
      {3, 3, 3, 1}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
      {1, 5, 5, 3},
      {13,  14,  15,  36,  39,  42,  26,  28,  30,  62,  67,  72,  39,
       42,  45,  38,  43,  48,  96,  108, 120, 64,  71,  78,  148, 164,
       180, 90,  99,  108, 52,  56,  60,  114, 123, 132, 65,  70,  75,
       140, 151, 162, 78,  84,  90,  116, 127, 138, 252, 276, 300, 142,
       155, 168, 304, 332, 360, 168, 183, 198, 91,  98,  105, 192, 207,
       222, 104, 112, 120, 218, 235, 252, 117, 126, 135});
}

template <DeviceType D>
void TestNHWCSimple3x3SAME_S2_1() {
  RunTestSimple<D>(
      {1, 3, 3, 1}, {12, 18, 12, 18, 27, 18, 12, 18, 12}, 2, Padding::SAME,
      {0, 0}, {1, 5, 5, 3}, {3, 3, 3, 1},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 5, 5, 3},
      {12, 12, 12, 30, 30, 30, 18, 18, 18, 30, 30, 30, 12, 12, 12,
       30, 30, 30, 75, 75, 75, 45, 45, 45, 75, 75, 75, 30, 30, 30,
       18, 18, 18, 45, 45, 45, 27, 27, 27, 45, 45, 45, 18, 18, 18,
       30, 30, 30, 75, 75, 75, 45, 45, 45, 75, 75, 75, 30, 30, 30,
       12, 12, 12, 30, 30, 30, 18, 18, 18, 30, 30, 30, 12, 12, 12});
}

template <DeviceType D>
void TestNHWCSimple3x3VALID_S2() {
  RunTestSimple<D>(
      {1, 3, 3, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, 2, Padding::VALID, {0, 0},
      {1, 7, 7, 3}, {3, 3, 3, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
      {1, 7, 7, 3},
      {1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
       2, 2, 2, 2, 2, 2, 4, 4, 4, 2, 2, 2, 4, 4, 4, 2, 2, 2, 2, 2, 2,
       1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
       2, 2, 2, 2, 2, 2, 4, 4, 4, 2, 2, 2, 4, 4, 4, 2, 2, 2, 2, 2, 2,
       1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1});
}

template <DeviceType D>
void TestNHWCSimple3x3VALID_S1() {
  RunTestSimple<D>(
      {1, 3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, Padding::VALID, {0, 0},
      {1, 5, 5, 3}, {3, 3, 3, 1},
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
       15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
      {1, 5, 5, 3},
      {1,   2,   3,   6,   9,   12,  18,  24,  30,  26,  31,  36,  21,
       24,  27,  14,  19,  24,  54,  66,  78,  126, 147, 168, 130, 146,
       162, 90,  99,  108, 66,  78,  90,  198, 225, 252, 405, 450, 495,
       366, 399, 432, 234, 252, 270, 146, 157, 168, 354, 378, 402, 630,
       669, 708, 502, 530, 558, 294, 309, 324, 133, 140, 147, 306, 321,
       336, 522, 546, 570, 398, 415, 432, 225, 234, 243});
}

template <DeviceType D>
void TestNHWCSimple2x2SAME() {
  RunTestSimple<D>({1, 2, 2, 1}, {1, 1, 1, 1}, 1, Padding::SAME, {0, 0},
                   {1, 2, 2, 1}, {3, 3, 1, 1},
                   {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                   {1, 2, 2, 1}, {4.f, 4.f, 4.f, 4.f});
}

template <DeviceType D>
void TestNHWCSimple2x2VALID() {
  RunTestSimple<D>(
      {1, 2, 2, 1}, {1, 1, 1, 1}, 2, Padding::VALID, {0, 0}, {1, 5, 5, 1},
      {3, 3, 1, 1}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {1, 5, 5, 1},
      {1.f, 1.f, 2.f, 1.f, 1.f, 1.f, 1.f, 2.f, 1.f, 1.f, 2.f, 2.f, 4.f,
       2.f, 2.f, 1.f, 1.f, 2.f, 1.f, 1.f, 1.f, 1.f, 2.f, 1.f, 1.f});
}
}  // namespace

TEST_F(Deconv2dOpTest, CPUSimple3X3PaddingSame_S1) {
  TestNHWCSimple3x3SAME_S1<DeviceType::CPU>();
}

TEST_F(Deconv2dOpTest, CPUSimple3X3PaddingSame_S2) {
  TestNHWCSimple3x3SAME_S2<DeviceType::CPU>();
}

TEST_F(Deconv2dOpTest, CPUSimple3X3PaddingSame_S2_1) {
  TestNHWCSimple3x3SAME_S2_1<DeviceType::CPU>();
}

TEST_F(Deconv2dOpTest, CPUSimple2X2PaddingSame) {
  TestNHWCSimple2x2SAME<DeviceType::CPU>();
}

TEST_F(Deconv2dOpTest, CPUSimple2X2PaddingValid) {
  TestNHWCSimple2x2VALID<DeviceType::CPU>();
}

TEST_F(Deconv2dOpTest, CPUSimple3X3PaddingValid_S1) {
  TestNHWCSimple3x3VALID_S1<DeviceType::CPU>();
}

TEST_F(Deconv2dOpTest, CPUSimple3X3PaddingValid_S2) {
  TestNHWCSimple3x3VALID_S2<DeviceType::CPU>();
}

TEST_F(Deconv2dOpTest, OPENCLSimple2X2PaddingSame) {
  TestNHWCSimple2x2SAME<DeviceType::GPU>();
}

TEST_F(Deconv2dOpTest, OPENCLSimple3X3PaddingSame_S1) {
  TestNHWCSimple3x3SAME_S1<DeviceType::GPU>();
}

TEST_F(Deconv2dOpTest, OPENCLSimple3X3PaddingSame_S2) {
  TestNHWCSimple3x3SAME_S2<DeviceType::GPU>();
}

TEST_F(Deconv2dOpTest, OPENCLSimple3X3PaddingSame_S2_1) {
  TestNHWCSimple3x3SAME_S2_1<DeviceType::GPU>();
}

TEST_F(Deconv2dOpTest, OPENCLSimple2X2PaddingValid) {
  TestNHWCSimple2x2VALID<DeviceType::GPU>();
}

TEST_F(Deconv2dOpTest, OPENCLSimple3X3PaddingValid_S1) {
  TestNHWCSimple3x3VALID_S1<DeviceType::GPU>();
}

TEST_F(Deconv2dOpTest, OPENCLSimple3X3PaddingValid_S2) {
  TestNHWCSimple3x3VALID_S2<DeviceType::GPU>();
}

namespace {
template <DeviceType D, typename T>
void TestComplexDeconvNxNS12(const int batch,
                             const std::vector<int> &shape,
                             const int stride) {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type, int padding) {
    // generate random input
    static unsigned int seed = time(NULL);
    int height = shape[0];
    int width = shape[1];
    int input_channels = shape[2];
    int output_channels = shape[3];

    OpsTestNet net;

    // Add input data
    net.AddRandomInput<D, T>("Input", {batch, height, width, input_channels});
    net.AddRandomInput<D, T>(
        "Filter", {output_channels, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<D, T>("Bias", {output_channels});
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    int out_h = 0;
    int out_w = 0;

    std::vector<int> paddings;
    std::vector<int> output_shape;

    if (padding < 0) {
      if (type == Padding::SAME) {
        out_h = (height - 1) * stride_h + 1;
        out_w = (width - 1) * stride_w + 1;
      } else {
        out_h = (height - 1) * stride_h + kernel_h;
        out_w = (width - 1) * stride_w + kernel_w;
      }
      output_shape.push_back(batch);
      output_shape.push_back(out_h);
      output_shape.push_back(out_w);
      output_shape.push_back(output_channels);
    } else {
      //      out_h = (height - 1) * stride + 1 + padding - kernel_h + 1;
      //      out_w = (width -1) * stride + 1 + padding - kernel_w + 1;
      paddings.push_back(padding);
      paddings.push_back(padding);
    }
    bool from_caffe = output_shape.size() != 4;
    // Construct graph
    OpDefBuilder("Deconv2D", "Deconv2dTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("padding_values", paddings)
        .AddIntsArg("output_shape", output_shape)
        .AddIntArg("from_caffe", from_caffe)
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

    OpDefBuilder("Deconv2D", "Deconv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("padding_values", paddings)
        .AddIntsArg("output_shape", output_shape)
        .AddIntArg("from_caffe", from_caffe)
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
    func(kernel_size, kernel_size, stride, stride, VALID, -1);
    func(kernel_size, kernel_size, stride, stride, SAME, -1);
    func(kernel_size, kernel_size, stride, stride, VALID, 1);
    func(kernel_size, kernel_size, stride, stride, VALID, 2);
  }
}
}  // namespace

TEST_F(Deconv2dOpTest, OPENCLAlignedDeconvNxNS12) {
  TestComplexDeconvNxNS12<DeviceType::GPU, float>(1, {32, 16, 16, 32}, 1);
  TestComplexDeconvNxNS12<DeviceType::GPU, float>(1, {32, 16, 16, 32}, 2);
}

TEST_F(Deconv2dOpTest, OPENCLAlignedDeconvNxNS34) {
  TestComplexDeconvNxNS12<DeviceType::GPU, float>(1, {32, 16, 16, 32}, 3);
  TestComplexDeconvNxNS12<DeviceType::GPU, float>(1, {32, 16, 16, 32}, 4);
}

TEST_F(Deconv2dOpTest, OPENCLUnalignedDeconvNxNS12) {
  TestComplexDeconvNxNS12<DeviceType::GPU, float>(1, {17, 113, 5, 7}, 1);
  TestComplexDeconvNxNS12<DeviceType::GPU, float>(1, {17, 113, 5, 7}, 2);
}

TEST_F(Deconv2dOpTest, OPENCLUnalignedDeconvNxNS34) {
  TestComplexDeconvNxNS12<DeviceType::GPU, float>(1, {17, 113, 5, 7}, 3);
  TestComplexDeconvNxNS12<DeviceType::GPU, float>(1, {17, 113, 5, 7}, 4);
}

TEST_F(Deconv2dOpTest, OPENCLUnalignedDeconvNxNMultiBatch) {
  TestComplexDeconvNxNS12<DeviceType::GPU, float>(3, {17, 13, 5, 7}, 1);
  TestComplexDeconvNxNS12<DeviceType::GPU, float>(5, {17, 13, 5, 7}, 2);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
