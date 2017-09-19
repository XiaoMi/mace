//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/conv_2d.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class DepthwiseConv2dOpTest : public OpsTestBase {};

TEST_F(DepthwiseConv2dOpTest, Simple_VALID) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2dTest")
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
  net.AddInputFromArray<float>(
      "Input", {1, 2, 2, 3},
      {1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12});
  net.AddInputFromArray<float>(
      "Filter", {2, 2, 2, 2},
      {1.0f, 5.0f, 9.0f, 13.0f,
       2.0f, 6.0f, 10.0f, 14.0f,
       3.0f, 7.0f, 11.0f, 15.0f,
       4.0f, 8.0f, 12.0f, 16.0f});
  net.AddInputFromArray<float>("Bias", {4}, {.1f, .2f, .3f, .4f});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 4, 1, 2},
                                      {196.1f, 252.1f, 216.2f, 280.2f,
                                      272.3f, 344.3f, 296.4f, 376.4f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(DepthwiseConv2dOpTest, ConvNxNS12) {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type) {
    srand(time(NULL));

    // generate random input
    index_t batch = 2 + rand() % 10;
    index_t input_channels = 3 + rand() % 10;
    index_t height = 107;
    index_t width = 113;
    index_t multiplier = 3 + rand() % 10;
    // Construct graph
    auto& net = test_net();
    OpDefBuilder("DepthwiseConv2d", "DepthwiseConv2dTest")
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
    net.AddRandomInput<float>("Input", {batch, input_channels, height, width});
    net.AddRandomInput<float>(
        "Filter", {multiplier, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<float>("Bias", {multiplier * input_channels});
    // run cpu
    net.RunOp();

    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // Run NEON
    net.RunOp(DeviceType::NEON);
    ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 1e-3);
  };

  for (int kernel_size : {3}) {
    for (int stride : {1, 2}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }
}
