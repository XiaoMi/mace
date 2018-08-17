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

#include "mace/ops/conv_2d.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class DepthwiseConv2dOpTest : public OpsTestBase {};

namespace {
template <DeviceType D, typename T>
void SimpleValidTest() {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 3, 3, 2},
      {1, 2, 2, 4, 3, 6, 4, 8, 5, 10, 6, 12, 7, 14, 8, 16, 9, 18});
  net.AddInputFromArray<D, float>(
      "Filter", {1, 2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 4.0f, 6.0f, 8.0f});
  net.AddInputFromArray<D, float>("Bias", {2}, {.1f, .2f});
  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
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
    BufferToImage<D, T>(&net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(&net, "Filter", "FilterImage",
                        kernels::BufferType::DW_CONV2D_FILTER);
    BufferToImage<D, T>(&net, "Bias", "BiasImage",
                        kernels::BufferType::ARGUMENT);
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
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
    ImageToBuffer<D, float>(&net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_CHANNEL);

  } else {
    MACE_NOT_IMPLEMENTED;
  }

  // Check
  auto expected = CreateTensor<float>(
      {1, 2, 2, 2},
      {37.1f, 148.2f, 47.1f, 188.2f, 67.1f, 268.2f, 77.1f, 308.2f});

  if (DataTypeToEnum<T>::value == DT_HALF) {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-3, 1e-3);
  } else {
    ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
  }
}
}  // namespace

TEST_F(DepthwiseConv2dOpTest, SimpleCPU) {
  SimpleValidTest<DeviceType::CPU, float>();
}

TEST_F(DepthwiseConv2dOpTest, SimpleOpenCL) {
  SimpleValidTest<DeviceType::GPU, float>();
}

TEST_F(DepthwiseConv2dOpTest, SimpleOpenCLHalf) {
  SimpleValidTest<DeviceType::GPU, half>();
}

namespace {
template <DeviceType D, typename T>
void ComplexValidTest(index_t batch,
                      index_t channel,
                      index_t height,
                      index_t width,
                      index_t kernel,
                      index_t multiplier,
                      int stride) {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  std::vector<float> input_data(batch * height * width * channel);
  GenerateRandomRealTypeData({batch, height, width, channel}, &input_data);
  net.AddInputFromArray<D, float>("Input", {batch, height, width, channel},
                                  input_data);
  std::vector<float> filter_data(kernel * kernel * channel * multiplier);
  GenerateRandomRealTypeData({multiplier, channel, kernel, kernel},
                             &filter_data);
  net.AddInputFromArray<D, float>(
      "Filter", {multiplier, channel, kernel, kernel}, filter_data);
  std::vector<float> bias_data(channel * multiplier);
  GenerateRandomRealTypeData({channel * multiplier}, &bias_data);
  net.AddInputFromArray<D, float>("Bias", {channel * multiplier}, bias_data);

  if (D == DeviceType::CPU) {
    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {stride, stride})
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
                        kernels::BufferType::DW_CONV2D_FILTER);
    BufferToImage<D, T>(&net, "Bias", "BiasImage",
                        kernels::BufferType::ARGUMENT);
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {stride, stride})
        .AddIntArg("padding", Padding::SAME)
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

  // expect
  index_t out_height = (height - 1) / stride + 1;
  index_t out_width = (width - 1) / stride + 1;
  index_t pad_top = ((out_height - 1) * stride + kernel - height) >> 1;
  index_t pad_left = ((out_width - 1) * stride + kernel - width) >> 1;
  index_t out_channels = channel * multiplier;
  std::vector<T> expect(batch * out_height * out_width * out_channels);
  for (index_t b = 0; b < batch; ++b) {
    for (index_t h = 0; h < out_height; ++h) {
      for (index_t w = 0; w < out_width; ++w) {
        for (index_t m = 0; m < out_channels; ++m) {
          index_t out_offset =
              ((b * out_height + h) * out_width + w) * out_channels + m;
          index_t c = m / multiplier;
          index_t o = m % multiplier;
          float sum = 0;
          for (index_t kh = 0; kh < kernel; ++kh) {
            for (index_t kw = 0; kw < kernel; ++kw) {
              index_t ih = h * stride - pad_top + kh;
              index_t iw = w * stride - pad_left + kw;
              if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                index_t in_offset =
                    ((b * height + ih) * width + iw) * channel + c;
                index_t filter_offset =
                    ((o * channel + c) * kernel + kh) * kernel + kw;
                sum += input_data[in_offset] * filter_data[filter_offset];
              }
            }
          }
          expect[out_offset] = sum + bias_data[m];
        }
      }
    }
  }

  auto expected =
      CreateTensor<T>({1, out_height, out_width, out_channels}, expect);

  if (DataTypeToEnum<T>::value == DT_FLOAT) {
    ExpectTensorNear<T>(*expected, *net.GetOutput("Output"), 1e-5);
  } else {
    ExpectTensorNear<T>(*expected, *net.GetOutput("Output"), 1e-2);
  }
}
}  // namespace

TEST_F(DepthwiseConv2dOpTest, ComplexCPU) {
  ComplexValidTest<DeviceType::CPU, float>(1, 3, 10, 10, 5, 1, 2);
}

TEST_F(DepthwiseConv2dOpTest, ComplexCPU3x3s1) {
  ComplexValidTest<DeviceType::CPU, float>(1, 3, 10, 10, 3, 1, 1);
}

TEST_F(DepthwiseConv2dOpTest, ComplexCPU3x3s2) {
  ComplexValidTest<DeviceType::CPU, float>(1, 3, 10, 10, 3, 1, 2);
}

TEST_F(DepthwiseConv2dOpTest, ComplexOpenCL) {
  ComplexValidTest<DeviceType::GPU, float>(1, 3, 10, 10, 5, 1, 2);
}

TEST_F(DepthwiseConv2dOpTest, ComplexOpenCLHalf) {
  ComplexValidTest<DeviceType::GPU, half>(1, 3, 10, 10, 5, 1, 2);
}

namespace {
template <typename T>
void TestNxNS12(const index_t height, const index_t width) {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type) {
    // generate random input
    static unsigned int seed = time(NULL);
    index_t batch = 1 + rand_r(&seed) % 5;
    index_t input_channels = 3 + rand_r(&seed) % 16;
    index_t multiplier = 1;
    // Construct graph
    OpsTestNet net;

    // Add input data
    net.AddRandomInput<DeviceType::GPU, float>(
        "Input", {batch, height, width, input_channels});
    net.AddRandomInput<DeviceType::GPU, float>(
        "Filter", {multiplier, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<DeviceType::GPU, float>("Bias",
                                               {multiplier * input_channels});

    net.TransformDataFormat<DeviceType::CPU, float>("Input", NHWC, "InputNCHW",
                                                    NCHW);
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
        .Input("InputNCHW")
        .Input("Filter")
        .Input("Bias")
        .Output("OutputNCHW")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<float>::value))
        .Finalize(net.NewOperatorDef());

    // Run on cpu
    net.RunOp();

    net.TransformDataFormat<DeviceType::CPU, float>("OutputNCHW", NCHW,
                                                    "Output", NHWC);

    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    BufferToImage<DeviceType::GPU, T>(&net, "Input", "InputImage",
                                      kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<DeviceType::GPU, T>(&net, "Filter", "FilterImage",
                                      kernels::BufferType::DW_CONV2D_FILTER);
    BufferToImage<DeviceType::GPU, T>(&net, "Bias", "BiasImage",
                                      kernels::BufferType::ARGUMENT);
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());

    net.RunOp(DeviceType::GPU);

    // Transfer output
    ImageToBuffer<DeviceType::GPU, float>(&net, "OutputImage", "DeviceOutput",
                                          kernels::BufferType::IN_OUT_CHANNEL);

    // Check
    if (DataTypeToEnum<T>::value == DT_FLOAT) {
      ExpectTensorNear<float>(expected, *net.GetOutput("DeviceOutput"), 1e-5,
                              1e-4);
    } else {
      ExpectTensorNear<float>(expected, *net.GetOutput("DeviceOutput"), 1e-2,
                              1e-2);
    }
  };

  for (int kernel_size : {2, 3, 4}) {
    for (int stride : {1, 2}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }
}
}  // namespace

TEST_F(DepthwiseConv2dOpTest, OpenCLSimpleNxNS12) { TestNxNS12<float>(4, 4); }

TEST_F(DepthwiseConv2dOpTest, OpenCLSimpleNxNS12Half) {
  TestNxNS12<half>(4, 4);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLAlignedNxNS12) {
  TestNxNS12<float>(128, 128);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLAlignedNxNS12Half) {
  TestNxNS12<half>(128, 128);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLUnalignedNxNS12) {
  TestNxNS12<float>(107, 113);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLUnalignedNxNS12Half) {
  TestNxNS12<half>(107, 113);
}

namespace {

void QuantSimpleValidTest() {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<CPU, uint8_t>(
      "Input", {1, 3, 3, 2},
      {31, 98, 1, 54, 197, 172, 70, 146, 255, 71, 24, 182, 28, 78, 85, 96, 180,
       59}, 0.00735299, 86);
  net.AddInputFromArray<CPU, uint8_t>(
      "Filter", {3, 3, 2, 1},
      {212, 239, 110, 170, 216, 91, 162, 161, 255, 2, 10, 120, 183, 101, 100,
       33, 137, 51}, 0.0137587, 120);
  net.AddInputFromArray<CPU, int32_t>("Bias", {2}, {2, 2});
  OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DT_UINT8))
      .Finalize(net.NewOperatorDef());

  net.Setup(CPU);
  Tensor *output = net.GetTensor("Output");
  output->SetScale(0.013241);
  output->SetZeroPoint(0);
  // Run
  net.Run();

  // Check
  auto expected = CreateTensor<uint8_t>({1, 1, 1, 2}, {255, 21});

  ExpectTensorNear<uint8_t>(*expected, *net.GetOutput("Output"));
}

void TestQuant(const index_t batch,
               const index_t multiplier,
               const index_t in_channels,
               const index_t in_height,
               const index_t in_width,
               const index_t k_height,
               const index_t k_width,
               enum Padding padding_type,
               const std::vector<int> &strides) {
  OpsTestNet net;
  const index_t out_channels = multiplier * in_channels;
  net.AddRandomInput<CPU, float>(
      "Input", {batch, in_height, in_width, in_channels}, false);
  net.AddRandomInput<CPU, float>(
      "Filter", {k_height, k_width, in_channels, multiplier}, false);
  net.AddRandomInput<CPU, float>("Bias", {out_channels});
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Input", NHWC, "InputNCHW", NCHW);
  net.TransformDataFormat<DeviceType::CPU, float>(
      "Filter", HWIO, "FilterOIHW", OIHW);

  OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
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
  net.TransformDataFormat<DeviceType::CPU, float>(
      "OutputNCHW", NCHW, "Output", NHWC);

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
  net.AddInputFromArray<DeviceType::CPU, int32_t>(
      "QuantizedBias", {out_channels}, q_bias);
  OpDefBuilder("DepthwiseConv2d", "QuantizedDepthwiseConv2DTest")
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

TEST_F(DepthwiseConv2dOpTest, Quant) {
  QuantSimpleValidTest();
  TestQuant(1, 1, 1024, 7, 7, 3, 3, VALID, {1, 1});
  TestQuant(1, 1, 1024, 7, 7, 3, 3, SAME, {1, 1});
  TestQuant(1, 1, 1024, 7, 7, 3, 3, FULL, {1, 1});
  TestQuant(1, 2, 1024, 7, 7, 3, 3, SAME, {1, 1});
  TestQuant(1, 2, 1024, 7, 7, 3, 3, SAME, {2, 2});
  TestQuant(1, 1, 512, 14, 14, 3, 3, SAME, {1, 1});
  TestQuant(1, 1, 512, 14, 13, 5, 5, SAME, {2, 2});
  TestQuant(1, 1, 256, 28, 28, 3, 3, SAME, {1, 1});
  TestQuant(1, 1, 128, 56, 56, 3, 3, SAME, {2, 2});
  TestQuant(3, 1, 128, 56, 56, 3, 3, SAME, {2, 2});
}

}  // namespace test
}  // namespace ops
}  // namespace mace
