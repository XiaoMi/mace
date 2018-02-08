//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/conv_2d.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

namespace {

class DepthwiseConv2dOpTest : public OpsTestBase {};

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
      "Filter", {2, 2, 2, 1}, {1.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f, 8.0f});
  net.AddInputFromArray<D, float>("Bias", {2}, {.1f, .2f});
  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(net, "Filter", "FilterImage",
                        kernels::BufferType::DW_CONV2D_FILTER);
    BufferToImage<D, T>(net, "Bias", "BiasImage",
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
    ImageToBuffer<D, T>(net, "OutputImage", "Output",
                        kernels::BufferType::IN_OUT_CHANNEL);

  } else {
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }

  // Check
  auto expected = CreateTensor<T>({1, 2, 2, 2}, {37.1f, 148.2f, 47.1f, 188.2f,
                                                 67.1f, 268.2f, 77.1f, 308.2f});

  ExpectTensorNear<T>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(DepthwiseConv2dOpTest, SimpleCPU) {
  SimpleValidTest<DeviceType::CPU, float>();
}

TEST_F(DepthwiseConv2dOpTest, SimpleOpenCL) {
  SimpleValidTest<DeviceType::OPENCL, float>();
}

TEST_F(DepthwiseConv2dOpTest, SimpleOpenCLHalf) {
  SimpleValidTest<DeviceType::OPENCL, half>();
}

template <DeviceType D, typename T>
void ComplexValidTest() {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 10, 10, 3},
      {0.0,  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,  0.11,
       0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,  0.21, 0.22, 0.23,
       0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,  0.31, 0.32, 0.33, 0.34, 0.35,
       0.36, 0.37, 0.38, 0.39, 0.4,  0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47,
       0.48, 0.49, 0.5,  0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
       0.6,  0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,  0.71,
       0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,  0.81, 0.82, 0.83,
       0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,  0.91, 0.92, 0.93, 0.94, 0.95,
       0.96, 0.97, 0.98, 0.99, 1.0,  1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07,
       1.08, 1.09, 1.1,  1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19,
       1.2,  1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3,  1.31,
       1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4,  1.41, 1.42, 1.43,
       1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5,  1.51, 1.52, 1.53, 1.54, 1.55,
       1.56, 1.57, 1.58, 1.59, 1.6,  1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67,
       1.68, 1.69, 1.7,  1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79,
       1.8,  1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.9,  1.91,
       1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 2.0,  2.01, 2.02, 2.03,
       2.04, 2.05, 2.06, 2.07, 2.08, 2.09, 2.1,  2.11, 2.12, 2.13, 2.14, 2.15,
       2.16, 2.17, 2.18, 2.19, 2.2,  2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27,
       2.28, 2.29, 2.3,  2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.39,
       2.4,  2.41, 2.42, 2.43, 2.44, 2.45, 2.46, 2.47, 2.48, 2.49, 2.5,  2.51,
       2.52, 2.53, 2.54, 2.55, 2.56, 2.57, 2.58, 2.59, 2.6,  2.61, 2.62, 2.63,
       2.64, 2.65, 2.66, 2.67, 2.68, 2.69, 2.7,  2.71, 2.72, 2.73, 2.74, 2.75,
       2.76, 2.77, 2.78, 2.79, 2.8,  2.81, 2.82, 2.83, 2.84, 2.85, 2.86, 2.87,
       2.88, 2.89, 2.9,  2.91, 2.92, 2.93, 2.94, 2.95, 2.96, 2.97, 2.98, 2.99});
  net.AddInputFromArray<D, float>(
      "Filter", {5, 5, 3, 1},
      {0.0,  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
       0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,  0.21,
       0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,  0.31, 0.32,
       0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,  0.41, 0.42, 0.43,
       0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,  0.51, 0.52, 0.53, 0.54,
       0.55, 0.56, 0.57, 0.58, 0.59, 0.6,  0.61, 0.62, 0.63, 0.64, 0.65,
       0.66, 0.67, 0.68, 0.69, 0.7,  0.71, 0.72, 0.73, 0.74});
  net.AddInputFromArray<D, float>("Bias", {6},
                                  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT_CHANNEL);
    BufferToImage<D, T>(net, "Filter", "FilterImage",
                        kernels::BufferType::DW_CONV2D_FILTER);
    BufferToImage<D, T>(net, "Bias", "BiasImage",
                        kernels::BufferType::ARGUMENT);
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());

    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, T>(net, "OutputImage", "Output",
                        kernels::BufferType::IN_OUT_CHANNEL);

  } else {
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }

  // Check
  auto expected = CreateTensor<T>(
      {1, 5, 5, 3},
      {4.48200035,  4.63479996,  4.79079962,  5.85899973,  6.05599976,
       6.25699997,  6.38100004,  6.59000015,  6.80300045,  6.90299988,
       7.1239996,   7.34899998,  4.03559971,  4.16820002,  4.30319977,
       8.90999985,  9.1760006,   9.44599915,  11.20499992, 11.54500103,
       11.89000034, 11.74499989, 12.09999943, 12.46000004, 12.28499985,
       12.65500069, 13.03000069, 7.00200033,  7.22399998,  7.44900036,
       13.4100008,  13.79599953, 14.18599987, 16.60500145, 17.09499741,
       17.59000015, 17.14500046, 17.65000153, 18.15999794, 17.68499947,
       18.20499992, 18.72999954, 9.97200012,  10.28399944, 10.59899998,
       17.90999985, 18.41600037, 18.92599869, 22.00500107, 22.64500046,
       23.28999901, 22.54500008, 23.19999886, 23.8599987,  23.0850029,
       23.75500107, 24.43000031, 12.94200039, 13.34400082, 13.7489996,
       6.97500038,  7.29659986,  7.62060022,  8.32049942,  8.72700024,
       9.13650036,  8.5095005,   8.92500019,  9.34349918,  8.69849968,
       9.12300014,  9.55049992,  4.55220032,  4.80690002,  5.06340027});

  ExpectTensorNear<T>(*expected, *net.GetOutput("Output"), 0.2);
}

TEST_F(DepthwiseConv2dOpTest, ComplexCPU) {
  ComplexValidTest<DeviceType::CPU, float>();
}

TEST_F(DepthwiseConv2dOpTest, ComplexOpenCL) {
  ComplexValidTest<DeviceType::OPENCL, float>();
}

TEST_F(DepthwiseConv2dOpTest, ComplexOpenCLHalf) {
  ComplexValidTest<DeviceType::OPENCL, half>();
}

template <DeviceType D, typename T>
void TestNxNS12(const index_t height, const index_t width) {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type) {
    srand(time(NULL));

    // generate random input
    index_t batch = 1 + rand() % 5;
    index_t input_channels = 3 + rand() % 16;
    index_t multiplier = 1;
    // Construct graph
    OpsTestNet net;

    // Add input data
    net.AddRandomInput<D, float>("Input",
                                 {batch, height, width, input_channels});
    net.AddRandomInput<D, float>(
        "Filter", {kernel_h, kernel_w, input_channels, multiplier});
    net.AddRandomInput<D, float>("Bias", {multiplier * input_channels});
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<float>::value))
        .Finalize(net.NewOperatorDef());

    // Run on cpu
    net.RunOp();
    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    if (D == DeviceType::OPENCL) {
      BufferToImage<D, T>(net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT_CHANNEL);
      BufferToImage<D, T>(net, "Filter", "FilterImage",
                          kernels::BufferType::DW_CONV2D_FILTER);
      BufferToImage<D, T>(net, "Bias", "BiasImage",
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

      net.RunOp(D);

      // Transfer output
      ImageToBuffer<D, float>(net, "OutputImage", "DeviceOutput",
                              kernels::BufferType::IN_OUT_CHANNEL);
    } else {
      OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
          .Input("Input")
          .Input("Filter")
          .Input("Bias")
          .Output("DeviceOutput")
          .AddIntsArg("strides", {stride_h, stride_w})
          .AddIntArg("padding", type)
          .AddIntsArg("dilations", {1, 1})
          .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
          .Finalize(net.NewOperatorDef());

      // Run
      net.RunOp(D);
    }

    // Check
    ExpectTensorNear<float>(expected, *net.GetOutput("DeviceOutput"), 0.1);
  };

  for (int kernel_size : {3}) {
    for (int stride : {1, 2}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }
}

TEST_F(DepthwiseConv2dOpTest, OpenCLSimpleNxNS12) {
  TestNxNS12<DeviceType::OPENCL, float>(4, 4);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLSimpleNxNS12Half) {
  TestNxNS12<DeviceType::OPENCL, half>(4, 4);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLAlignedNxNS12) {
  TestNxNS12<DeviceType::OPENCL, float>(64, 64);
  TestNxNS12<DeviceType::OPENCL, float>(128, 128);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLAlignedNxNS12Half) {
  TestNxNS12<DeviceType::OPENCL, half>(64, 64);
  TestNxNS12<DeviceType::OPENCL, half>(128, 128);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLUnalignedNxNS12) {
  TestNxNS12<DeviceType::OPENCL, float>(107, 113);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLUnalignedNxNS12Half) {
  TestNxNS12<DeviceType::OPENCL, half>(107, 113);
}

}  // namespace
