//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <fstream>
#include "mace/ops/conv_2d.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class Conv2dOpTest : public OpsTestBase {};

template <DeviceType D>
void TestSimple3x3VALID() {
  OpsTestNet net;
  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add args

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 2, 3, 3},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, float>(
      "Filter", {1, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<D, float>("Bias", {1}, {0.1f});

  // Run
  net.RunOp(D);

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 1}, {18.1f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

template <DeviceType D>
void TestSimple3x3SAME() {
  OpsTestNet net;
  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 2, 3, 3},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, float>(
      "Filter", {1, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<D, float>("Bias", {1}, {0.1f});

  // Run
  net.RunOp(D);

  // Check
  auto expected = CreateTensor<float>(
      {1, 1, 3, 3},
      {8.1f, 12.1f, 8.1f, 12.1f, 18.1f, 12.1f, 8.1f, 12.1f, 8.1f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

#if __ARM_NEON
TEST_F(Conv2dOpTest, NEONSimple) {
  TestSimple3x3VALID<DeviceType::NEON>();
  TestSimple3x3SAME<DeviceType::NEON>();
}
#endif

template <DeviceType D, typename T>
void TestNHWCSimple3x3VALID() {
  OpsTestNet net;
  // Add input data
  net.AddInputFromArray<D, T>(
      "Input", {1, 3, 3, 2},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, T>(
      "Filter", {3, 3, 2, 1},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<D, T>("Bias", {1}, {0.1f});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT);
    BufferToImage<D, T>(net, "Filter", "FilterImage",
                        kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, T>(net, "Bias", "BiasImage",
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
    ImageToBuffer<D, T>(net, "OutputImage", "Output",
                        kernels::BufferType::IN_OUT);

  } else {
    OpDefBuilder("Conv2D", "Conv2dTest")
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

  auto expected = CreateTensor<float>({1, 1, 1, 1}, {18.1f});
  ExpectTensorNear<float, T>(*expected, *net.GetOutput("Output"), 0.01);
}

template <DeviceType D, typename T>
void TestNHWCSimple3x3SAME() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>(
      "Input", {1, 3, 3, 2},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, T>(
      "Filter", {3, 3, 2, 1},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<D, T>("Bias", {1}, {0.1f});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT);
    BufferToImage<D, T>(net, "Filter", "FilterImage",
                        kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, T>(net, "Bias", "BiasImage",
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
    ImageToBuffer<D, T>(net, "OutputImage", "Output",
                        kernels::BufferType::IN_OUT);

  } else {
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }

  auto expected = CreateTensor<float>(
      {1, 3, 3, 1},
      {8.1f, 12.1f, 8.1f, 12.1f, 18.1f, 12.1f, 8.1f, 12.1f, 8.1f});

  ExpectTensorNear<float, T>(*expected, *net.GetOutput("Output"), 0.01);
}

TEST_F(Conv2dOpTest, CPUSimple) {
  TestNHWCSimple3x3VALID<DeviceType::CPU, float>();
  TestNHWCSimple3x3SAME<DeviceType::CPU, float>();
}

TEST_F(Conv2dOpTest, OPENCLSimple) {
  TestNHWCSimple3x3VALID<DeviceType::OPENCL, float>();
  TestNHWCSimple3x3SAME<DeviceType::OPENCL, float>();
}

template <DeviceType D>
void TestSimple3x3WithoutBias() {
  OpsTestNet net;
  OpDefBuilder("Conv2D", "Conv2dTest")
      .Input("Input")
      .Input("Filter")
      .Output("Output")
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 2, 3, 3},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, float>(
      "Filter", {1, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

  // Run
  net.RunOp(D);

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 1}, {18.0f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

#ifdef __ARM_NEON
TEST_F(Conv2dOpTest, NEONWithouBias) {
  TestSimple3x3WithoutBias<DeviceType::NEON>();
}
#endif

template <DeviceType D, typename T>
void TestNHWCSimple3x3WithoutBias() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>(
      "Input", {1, 3, 3, 2},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, T>(
      "Filter", {3, 3, 2, 1},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT);
    BufferToImage<D, T>(net, "Filter", "FilterImage",
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
    ImageToBuffer<D, T>(net, "OutputImage", "Output",
                        kernels::BufferType::IN_OUT);
  } else {
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
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
  auto expected = CreateTensor<float>({1, 1, 1, 1}, {18.0f});

  ExpectTensorNear<float, T>(*expected, *net.GetOutput("Output"), 0.01);
}

TEST_F(Conv2dOpTest, CPUWithoutBias) {
  TestNHWCSimple3x3WithoutBias<DeviceType::CPU, float>();
}

TEST_F(Conv2dOpTest, OPENCLWithoutBias) {
  TestNHWCSimple3x3WithoutBias<DeviceType::OPENCL, float>();
}

template <DeviceType D>
static void TestCombined3x3() {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Conv2D", "Conv2DTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 2, 5, 5}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, float>(
      "Filter", {2, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
       0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
  net.AddInputFromArray<D, float>("Bias", {2}, {0.1f, 0.2f});

  // Run
  net.RunOp(D);

  // Check
  auto expected = CreateTensor<float>(
      {1, 2, 3, 3}, {8.1f, 12.1f, 8.1f, 12.1f, 18.1f, 12.1f, 8.1f, 12.1f, 8.1f,
                     4.2f, 6.2f, 4.2f, 6.2f, 9.2f, 6.2f, 4.2f, 6.2f, 4.2f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

#ifdef __ARM_NEON
TEST_F(Conv2dOpTest, NEONCombined) { TestCombined3x3<DeviceType::NEON>(); }
#endif

template <DeviceType D, typename T>
static void TestNHWCCombined3x3() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, T>(
      "Input", {1, 5, 5, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, T>(
      "Filter", {3, 3, 2, 2},
      {1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f,
       1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f,
       1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f});
  net.AddInputFromArray<D, T>("Bias", {2}, {0.1f, 0.2f});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, T>(net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT);
    BufferToImage<D, T>(net, "Filter", "FilterImage",
                        kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, T>(net, "Bias", "BiasImage",
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

    ImageToBuffer<D, T>(net, "OutputImage", "Output",
                        kernels::BufferType::IN_OUT);
  } else {
    OpDefBuilder("Conv2D", "Conv2DTest")
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
  auto expected = CreateTensor<float>(
      {1, 3, 3, 2}, {8.1f, 4.2f, 12.1f, 6.2f, 8.1f, 4.2f, 12.1f, 6.2f, 18.1f,
                     9.2f, 12.1f, 6.2f, 8.1f, 4.2f, 12.1f, 6.2f, 8.1f, 4.2f});
  ExpectTensorNear<float, T>(*expected, *net.GetOutput("Output"), 0.01);
}

TEST_F(Conv2dOpTest, CPUStride2) {
  TestNHWCCombined3x3<DeviceType::CPU, float>();
}

TEST_F(Conv2dOpTest, OPENCLStride2) {
  TestNHWCCombined3x3<DeviceType::OPENCL, float>();
}

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
      "Filter", {1, 1, 5, 2},
      {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f});
  net.AddInputFromArray<D, float>("Bias", {2}, {0.1f, 0.2f});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT);
    BufferToImage<D, float>(net, "Filter", "FilterImage",
                            kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, float>(net, "Bias", "BiasImage",
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

    ImageToBuffer<D, float>(net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT);
  } else {
    OpDefBuilder("Conv2D", "Conv2DTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
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

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(Conv2dOpTest, CPUConv1x1) { TestConv1x1<DeviceType::CPU>(); }

TEST_F(Conv2dOpTest, OPENCLConv1x1) { TestConv1x1<DeviceType::OPENCL>(); }

template <DeviceType D, typename T>
static void TestComplexConvNxNS12(const std::vector<index_t> &shape) {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type) {
    srand(time(NULL));

    // generate random input
    index_t batch = 3 + (rand() % 10);
    index_t height = shape[0];
    index_t width = shape[1];
    index_t input_channels = shape[2] + (rand() % 10);
    index_t output_channels = shape[3] + (rand() % 10);
    // Construct graph
    OpsTestNet net;
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", {1, 1})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());

    // Add input data
    net.AddRandomInput<D, T>("Input", {batch, height, width, input_channels});
    net.AddRandomInput<D, T>(
        "Filter", {kernel_h, kernel_w, input_channels, output_channels});
    net.AddRandomInput<D, T>("Bias", {output_channels});

    // run on cpu
    net.RunOp();
    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run on gpu
    BufferToImage<D, T>(net, "Input", "InputImage",
                        kernels::BufferType::IN_OUT);
    BufferToImage<D, T>(net, "Filter", "FilterImage",
                        kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, T>(net, "Bias", "BiasImage",
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

    ImageToBuffer<D, T>(net, "OutputImage", "OPENCLOutput",
                        kernels::BufferType::IN_OUT);
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 0.001);
  };

  for (int kernel_size : {1, 3}) {
    for (int stride : {1, 2}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }
}

TEST_F(Conv2dOpTest, OPENCLAlignedConvNxNS12) {
  TestComplexConvNxNS12<DeviceType::OPENCL, float>({32, 32, 32, 64});
}

TEST_F(Conv2dOpTest, OPENCLUnalignedConvNxNS12) {
  TestComplexConvNxNS12<DeviceType::OPENCL, float>({107, 113, 5, 7});
}

template<DeviceType D>
static void TestHalfComplexConvNxNS12(const std::vector<index_t> &input_shape,
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
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", padding)
        .AddIntsArg("dilations", {dilations[0], dilations[1]})
        .Finalize(net.NewOperatorDef());

    std::vector<float> float_input_data;
    GenerateRandomRealTypeData({batch, height, width, input_channels},
                               float_input_data);
    std::vector<float> float_filter_data;
    GenerateRandomRealTypeData(
        {kernel_h, kernel_w, input_channels, output_channels},
        float_filter_data);
    std::vector<float> float_bias_data;
    GenerateRandomRealTypeData({output_channels}, float_bias_data);
    // Add input data
    net.AddInputFromArray<D, float>(
        "Input", {batch, height, width, input_channels}, float_input_data);
    net.AddInputFromArray<D, float>(
        "Filter", {kernel_h, kernel_w, input_channels, output_channels},
        float_filter_data);
    net.AddInputFromArray<D, float>("Bias", {output_channels}, float_bias_data);

    // run on cpu
    net.RunOp();
    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run on gpu
    BufferToImage<D, half>(net, "Input", "InputImage",
                           kernels::BufferType::IN_OUT);
    BufferToImage<D, half>(net, "Filter", "FilterImage",
                           kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, half>(net, "Bias", "BiasImage",
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

    ImageToBuffer<D, float>(net, "OutputImage", "OPENCLOutput",
                            kernels::BufferType::IN_OUT);

    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 0.5);
  };

  func(1, 1, VALID);
  func(1, 1, SAME);
  if (dilations[0] == 1) {
    func(2, 2, VALID);
    func(2, 2, SAME);
  }
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv1x1S12) {
  TestHalfComplexConvNxNS12<DeviceType::OPENCL>({32, 32},
                                                {1, 1, 32, 64},
                                                {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv3x3S12) {
  TestHalfComplexConvNxNS12<DeviceType::OPENCL>({32, 32},
                                                {3, 3, 32, 64},
                                                {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv15x1S12) {
  TestHalfComplexConvNxNS12<DeviceType::OPENCL>({32, 32},
                                                {15, 1, 256, 2},
                                                {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv1x15S12) {
  TestHalfComplexConvNxNS12<DeviceType::OPENCL>({32, 32},
                                                {1, 15, 256, 2},
                                                {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfAlignedConv7x75S12) {
  TestHalfComplexConvNxNS12<DeviceType::OPENCL>({32, 32},
                                                {7, 7, 3, 64},
                                                {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfUnalignedConv1x1S12) {
  TestHalfComplexConvNxNS12<DeviceType::OPENCL>({107, 113},
                                                {1, 1, 5, 7},
                                                {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfUnalignedConv3x3S12) {
  TestHalfComplexConvNxNS12<DeviceType::OPENCL>({107, 113},
                                                {3, 3, 5, 7},
                                                {1, 1});
}

TEST_F(Conv2dOpTest, OPENCLHalfConv5x5Dilation2) {
  TestHalfComplexConvNxNS12<DeviceType::OPENCL>({64, 64},
                                                {5, 5, 16, 16},
                                                {2, 2});
}

TEST_F(Conv2dOpTest, OPENCLHalfConv7x7Dilation2) {
  TestHalfComplexConvNxNS12<DeviceType::OPENCL>({64, 64},
                                                {7, 7, 16, 16},
                                                {2, 2});
}

TEST_F(Conv2dOpTest, OPENCLHalfConv7x7Dilation4) {
  TestHalfComplexConvNxNS12<DeviceType::OPENCL>({63, 67},
                                                {7, 7, 16, 16},
                                                {4, 4});
}

template<DeviceType D, typename T>
static void TestDilationConvNxN(const std::vector<index_t> &shape, const int dilation_rate) {
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
    // Construct graph
    OpsTestNet net;
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", {dilation_rate, dilation_rate})
        .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
        .Finalize(net.NewOperatorDef());

    // Add input data
    net.AddRandomInput<D, T>("Input", {batch, height, width, input_channels});
    net.AddRandomInput<D, T>(
        "Filter", {kernel_h, kernel_w, input_channels, output_channels});
    net.AddRandomInput<D, T>("Bias", {output_channels});

    // run on cpu
    net.RunOp();
    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run on gpu
    BufferToImage<D, T>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);
    BufferToImage<D, T>(net, "Filter", "FilterImage", kernels::BufferType::CONV2D_FILTER);
    BufferToImage<D, T>(net, "Bias", "BiasImage", kernels::BufferType::ARGUMENT);

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

    ImageToBuffer<D, T>(net, "OutputImage", "OPENCLOutput", kernels::BufferType::IN_OUT);
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 0.001);
  };

  for (int kernel_size : {3}) {
    for (int stride : {1}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }
}

TEST_F(Conv2dOpTest, OPENCLAlignedDilation2) {
  TestDilationConvNxN<DeviceType::OPENCL, float>({32, 32, 32, 64},
                                                 2);
}

TEST_F(Conv2dOpTest, OPENCLAligned2Dilation4) {
  TestDilationConvNxN<DeviceType::OPENCL, float>({128, 128, 16, 16},
                                                 4);
}

TEST_F(Conv2dOpTest, OPENCLUnalignedDilation4) {
  TestDilationConvNxN<DeviceType::OPENCL, float>({107, 113, 5, 7},
                                                 4);
}

