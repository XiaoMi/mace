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

}  // namespace test
}  // namespace ops
}  // namespace mace
