//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/conv_2d.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class Conv2dOpTest : public OpsTestBase {};

template<DeviceType D>
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

template<DeviceType D>
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

TEST_F(Conv2dOpTest, CPUSimple) {
  TestSimple3x3VALID<DeviceType::CPU>();
  TestSimple3x3SAME<DeviceType::CPU>();
}

TEST_F(Conv2dOpTest, NEONSimple) {
  TestSimple3x3VALID<DeviceType::NEON>();
  TestSimple3x3SAME<DeviceType::NEON>();
}

TEST_F(Conv2dOpTest, OPENCLSimple) {
  TestSimple3x3VALID<DeviceType::OPENCL>();
  TestSimple3x3SAME<DeviceType::OPENCL>();
}

template<DeviceType D>
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

  // Add args

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

TEST_F(Conv2dOpTest, CPUWithoutBias) {
  TestSimple3x3WithoutBias<DeviceType::CPU>();
}

TEST_F(Conv2dOpTest, NEONWithouBias) {
  TestSimple3x3WithoutBias<DeviceType::NEON>();
}

TEST_F(Conv2dOpTest, OPENCLWithoutBias) {
  TestSimple3x3WithoutBias<DeviceType::OPENCL>();
}

template<DeviceType D>
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

TEST_F(Conv2dOpTest, CPUCombined) {
  TestCombined3x3<DeviceType::CPU>();
}

TEST_F(Conv2dOpTest, NEONCombined) {
  TestCombined3x3<DeviceType::NEON>();
}

TEST_F(Conv2dOpTest, OPENCLCombined) {
  TestCombined3x3<DeviceType::OPENCL>();
}

template<DeviceType D>
void TestConv1x1() {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Conv2D", "Conv2DTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 5, 3, 10},
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

  // Run
  net.RunOp(D);

  // Check
  auto expected = CreateTensor<float>(
      {1, 2, 3, 10},
      {5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f,
       5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f,
       5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f,
       10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f,
       10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f,
       10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(Conv2dOpTest, CPUConv1x1) {
  TestConv1x1<DeviceType::CPU>();
}

TEST_F(Conv2dOpTest, OPENCLConv1x1) {
  TestConv1x1<DeviceType::OPENCL>();
}

template<DeviceType D>
static void TestAlignedConvNxNS12() {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type) {
    srand(time(NULL));

    // generate random input
    index_t batch = 3;
    index_t input_channels = 64;
    index_t height = 32;
    index_t width = 32;
    index_t output_channels = 128;
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
        .Finalize(net.NewOperatorDef());

    // Add input data
    net.AddRandomInput<D, float>("Input", {batch, input_channels, height, width});
    net.AddRandomInput<D, float>(
        "Filter", {output_channels, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<D, float>("Bias", {output_channels});
    // Run on device
    net.RunOp(D);

    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run cpu
    net.RunOp();
    ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 0.001);
  };

  for (int kernel_size : {1, 3, 5}) {
    for (int stride : {1, 2}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }
}

TEST_F(Conv2dOpTest, NEONAlignedConvNxNS12) {
  TestAlignedConvNxNS12<DeviceType::NEON>();
}

TEST_F(Conv2dOpTest, OPENCLAlignedConvNxNS12) {
  TestAlignedConvNxNS12<DeviceType::OPENCL>();
}

template<DeviceType D>
static void TestUnalignedConvNxNS12() {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type) {
    srand(time(NULL));

    // generate random input
    index_t batch = 3 + rand() % 10;
    index_t input_channels = 3 + rand() % 10;
    index_t height = 107;
    index_t width = 113;
    index_t output_channels = 3 + rand() % 10;
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
        .Finalize(net.NewOperatorDef());

    // Add input data
    net.AddRandomInput<D, float>("Input", {batch, input_channels, height, width});
    net.AddRandomInput<D, float>(
        "Filter", {output_channels, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<D, float>("Bias", {output_channels});
    // Run on device
    net.RunOp(D);

    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run cpu
    net.RunOp();
    ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 0.001);
  };

  for (int kernel_size : {1, 3, 5}) {
    for (int stride : {1, 2}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }
}

TEST_F(Conv2dOpTest, NEONUnalignedConvNxNS12) {
  TestUnalignedConvNxNS12<DeviceType::NEON>();
}

TEST_F(Conv2dOpTest, OPENCLUnalignedConvNxNS12) {
  TestUnalignedConvNxNS12<DeviceType::OPENCL>();
}
