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


TEST_F(Conv2dOpTest, NEONSimple) {
  TestSimple3x3VALID<DeviceType::NEON>();
  TestSimple3x3SAME<DeviceType::NEON>();
}

template<DeviceType D, typename T>
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
    BufferToImage<D>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);
    BufferToImage<D>(net, "Filter", "FilterImage", kernels::BufferType::FILTER);
    BufferToImage<D>(net, "Bias", "BiasImage", kernels::BufferType::ARGUMENT);
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());

    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D>(net, "OutputImage", "Output", kernels::BufferType::IN_OUT);

  } else {
    OpDefBuilder("Conv2D", "Conv2dTest")
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

  auto expected = CreateTensor<T>({1, 1, 1, 1}, {18.1f});
  ExpectTensorNear<T>(*expected, *net.GetOutput("Output"), 0.001);
}

template<DeviceType D, typename T>
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
    BufferToImage<D>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);
    BufferToImage<D>(net, "Filter", "FilterImage", kernels::BufferType::FILTER);
    BufferToImage<D>(net, "Bias", "BiasImage", kernels::BufferType::ARGUMENT);
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D>(net, "OutputImage", "Output", kernels::BufferType::IN_OUT);

  } else {
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }

  auto expected = CreateTensor<T>(
      {1, 3, 3, 1},
      {8.1f, 12.1f, 8.1f, 12.1f, 18.1f, 12.1f, 8.1f, 12.1f, 8.1f});

  ExpectTensorNear<T>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(Conv2dOpTest, CPUSimple) {
  TestNHWCSimple3x3VALID<DeviceType::CPU, float>();
  TestNHWCSimple3x3SAME<DeviceType::CPU, float>();
}

TEST_F(Conv2dOpTest, OPENCLSimple) {
  TestNHWCSimple3x3VALID<DeviceType::OPENCL, float>();
  TestNHWCSimple3x3SAME<DeviceType::OPENCL, float>();
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


TEST_F(Conv2dOpTest, NEONWithouBias) {
  TestSimple3x3WithoutBias<DeviceType::NEON>();
}

template<DeviceType D>
void TestNHWCSimple3x3WithoutBias() {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 3, 3, 2},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, float>(
      "Filter", {3, 3, 2, 1},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);
    BufferToImage<D>(net, "Filter", "FilterImage", kernels::BufferType::FILTER);

    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
    // Transfer output
    ImageToBuffer<D>(net, "OutputImage", "Output", kernels::BufferType::IN_OUT);
  } else {
    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Output("Output")
        .AddIntsArg("strides", {1, 1})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());

    // Run
    net.RunOp(D);
  }

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 1}, {18.0f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(Conv2dOpTest, CPUWithoutBias) {
  TestNHWCSimple3x3WithoutBias<DeviceType::CPU>();
}

TEST_F(Conv2dOpTest, OPENCLWithoutBias) {
  TestNHWCSimple3x3WithoutBias<DeviceType::OPENCL>();
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


TEST_F(Conv2dOpTest, NEONCombined) {
  TestCombined3x3<DeviceType::NEON>();
}

template<DeviceType D>
static void TestNHWCCombined3x3() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 5, 5, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<D, float>(
      "Filter", {3, 3, 2, 2},
      {1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f,
       1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f,
       1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f, 0.5f});
  net.AddInputFromArray<D, float>("Bias", {2}, {0.1f, 0.2f});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);
    BufferToImage<D>(net, "Filter", "FilterImage", kernels::BufferType::FILTER);
    BufferToImage<D>(net, "Bias", "BiasImage", kernels::BufferType::ARGUMENT);

    OpDefBuilder("Conv2D", "Conv2DTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    ImageToBuffer<D>(net, "OutputImage", "Output", kernels::BufferType::IN_OUT);
  } else {
    OpDefBuilder("Conv2D", "Conv2DTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::SAME)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

  }

  // Check
  auto expected = CreateTensor<float>(
      {1, 3, 3, 2}, {8.1f, 4.2f, 12.1f, 6.2f, 8.1f, 4.2f,
                     12.1f, 6.2f, 18.1f, 9.2f, 12.1f, 6.2f,
                     8.1f, 4.2f, 12.1f, 6.2f, 8.1f, 4.2f});
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);

}

TEST_F(Conv2dOpTest, CPUCombined) {
  TestNHWCCombined3x3<DeviceType::CPU>();
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

  // Run
  net.RunOp(D);

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

TEST_F(Conv2dOpTest, CPUConv1x1) {
  TestConv1x1<DeviceType::CPU>();
}

//TEST_F(Conv2dOpTest, OPENCLConv1x1) {
//  TestConv1x1<DeviceType::OPENCL>();
//}

template<DeviceType D>
static void TestComplexConvNxNS12(const std::vector<index_t> &shape) {
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
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());

    // Add input data
    net.AddRandomInput<D, float>("Input", {batch, height, width, input_channels});
    net.AddRandomInput<D, float>(
        "Filter", {kernel_h, kernel_w, input_channels, output_channels});
    net.AddRandomInput<D, float>("Bias", {output_channels});

    // run on cpu
    net.RunOp();
    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run on gpu
    BufferToImage<D>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);
    BufferToImage<D>(net, "Filter", "FilterImage", kernels::BufferType::FILTER);
    BufferToImage<D>(net, "Bias", "BiasImage", kernels::BufferType::ARGUMENT);

    OpDefBuilder("Conv2D", "Conv2dTest")
        .Input("InputImage")
        .Input("FilterImage")
        .Input("BiasImage")
        .Output("OutputImage")
        .AddIntsArg("strides", {stride_h, stride_w})
        .AddIntArg("padding", type)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    // Run on device
    net.RunOp(D);

    ImageToBuffer<D>(net, "OutputImage", "OPENCLOutput", kernels::BufferType::IN_OUT);
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 0.001);
  };

  for (int kernel_size : {3}) {
    for (int stride : {1}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }
}

TEST_F(Conv2dOpTest, OPENCLAlignedConvNxNS12) {
  TestComplexConvNxNS12<DeviceType::OPENCL>({32, 32, 64, 128});
}

TEST_F(Conv2dOpTest, OPENCLUnalignedConvNxNS12) {
  TestComplexConvNxNS12<DeviceType::OPENCL>({107, 113, 5, 7});
}
