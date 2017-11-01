//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/conv_2d.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class DepthwiseConv2dOpTest : public OpsTestBase {};

template <DeviceType D>
void SimpleValidTest() {
  testing::internal::LogToStderr();
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add args
  net.AddIntsArg("strides", {1, 1});
  net.AddIntArg("padding", Padding::VALID);
  net.AddIntsArg("dilations", {1, 1});

  // Add input data
  net.AddInputFromArray<D, float>("Input", {1, 2, 2, 3},
                                                {1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12});
  net.AddInputFromArray<D, float>(
      "Filter", {2, 2, 2, 2},
      {1.0f, 5.0f, 9.0f, 13.0f, 2.0f, 6.0f, 10.0f, 14.0f, 3.0f, 7.0f, 11.0f,
       15.0f, 4.0f, 8.0f, 12.0f, 16.0f});
  net.AddInputFromArray<D, float>("Bias", {4}, {.1f, .2f, .3f, .4f});
  // Run
  net.RunOp(D);

  // Check
  auto expected = CreateTensor<float>(
      {1, 4, 1, 2},
      {196.1f, 252.1f, 216.2f, 280.2f, 272.3f, 344.3f, 296.4f, 376.4f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);

}

TEST_F(DepthwiseConv2dOpTest, SimpleCPU) {
  SimpleValidTest<DeviceType::CPU>();
}

template <DeviceType D>
void TestNxNS12(const index_t height, const index_t width) {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type) {
    srand(time(NULL));

    // generate random input
    index_t batch = 1;
    index_t input_channels = 3;
    index_t multiplier = 2;
    // Construct graph
    OpsTestNet net;
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2DTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .Finalize(net.operator_def());

    // Add args
    net.AddIntsArg("strides", {stride_h, stride_w});
    net.AddIntArg("padding", type);
    net.AddIntsArg("dilations", {1, 1});

    // Add input data
    net.AddRandomInput<D, float>("Input", {batch, input_channels, height, width});
    net.AddRandomInput<D, float>("Filter", {multiplier, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<D, float>("Bias", {multiplier * input_channels});
    // Run on device
    net.RunOp(D);

    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // run cpu
    net.RunOp();
    ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 1e-3);
  };

  for (int kernel_size : {3}) {
    for (int stride : {1, 2}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }

}

TEST_F(DepthwiseConv2dOpTest, NeonSimpleNxNS12) {
  TestNxNS12<DeviceType::NEON>(4, 4);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLSimpleNxNS12) {
  TestNxNS12<DeviceType::OPENCL>(4, 4);
}

TEST_F(DepthwiseConv2dOpTest, NeonAlignedNxNS12) {
  TestNxNS12<DeviceType::NEON>(64, 64);
  TestNxNS12<DeviceType::NEON>(128, 128);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLAlignedNxNS12) {
  TestNxNS12<DeviceType::OPENCL>(64, 64);
  TestNxNS12<DeviceType::OPENCL>(128, 128);
}

TEST_F(DepthwiseConv2dOpTest, NeonUnalignedNxNS12) {
  TestNxNS12<DeviceType::NEON>(107, 113);
}

TEST_F(DepthwiseConv2dOpTest, OpenCLUnalignedNxNS12) {
  TestNxNS12<DeviceType::OPENCL>(107, 113);
}
