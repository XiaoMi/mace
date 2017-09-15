//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/conv_2d.h"
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class Conv2dOpTest : public OpsTestBase {};

TEST_F(Conv2dOpTest, Simple_VALID) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Conv2d", "Conv2dTest")
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
      "Input", {1, 2, 3, 3},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<float>(
      "Filter", {1, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<float>("Bias", {1}, {0.1f});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 1}, {18.1f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(Conv2dOpTest, Simple_SAME) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Conv2d", "Conv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add args
  net.AddIntsArg("strides", {1, 1});
  net.AddIntArg("padding", Padding::SAME);
  net.AddIntsArg("dilations", {1, 1});

  // Add input data
  net.AddInputFromArray<float>(
      "Input", {1, 2, 3, 3},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<float>(
      "Filter", {1, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<float>("Bias", {1}, {0.1f});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>(
      {1, 1, 3, 3},
      {8.1f, 12.1f, 8.1f, 12.1f, 18.1f, 12.1f, 8.1f, 12.1f, 8.1f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(Conv2dOpTest, Combined) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Conv2d", "Conv2dTest")
      .Input("Input")
      .Input("Filter")
      .Input("Bias")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add args
  net.AddIntsArg("strides", {2, 2});
  net.AddIntArg("padding", Padding::SAME);
  net.AddIntsArg("dilations", {1, 1});

  // Add input data
  net.AddInputFromArray<float>(
      "Input", {1, 2, 5, 5}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<float>(
      "Filter", {2, 2, 3, 3},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
       0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
  net.AddInputFromArray<float>("Bias", {2}, {0.1f, 0.2f});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>(
      {1, 2, 3, 3}, {8.1f, 12.1f, 8.1f, 12.1f, 18.1f, 12.1f, 8.1f, 12.1f, 8.1f,
                     4.2f, 6.2f, 4.2f, 6.2f, 9.2f, 6.2f, 4.2f, 6.2f, 4.2f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(Conv2dOpTest, Conv1x1) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Conv2d", "Conv2dTest")
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
      "Input", {1, 5, 3, 10},
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<float>(
      "Filter", {2, 5, 1, 1},
      {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});
  net.AddInputFromArray<float>("Bias", {2}, {0.1f, 0.2f});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>(
      {1, 2, 3, 10},
      {5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,
       5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,
       5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,  5.1f,
       10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f,
       10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f,
       10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

// TODO we need more tests
TEST_F(Conv2dOpTest, ConvNxNS12) {
  testing::internal::LogToStderr();
  auto func = [&](int kernel_h, int kernel_w, int stride_h, int stride_w,
                  Padding type) {
    srand(time(NULL));

    // generate random input
    index_t batch = 1 + rand() % 10;
    index_t input_channels = 1 + rand() % 50;
    index_t height = 11 + rand() % 100;
    index_t width = 11 + rand() % 100;
    index_t output_channels = 1 + rand() % 50;
    // Construct graph
    auto& net = test_net();
    OpDefBuilder("Conv2d", "Conv2dTest")
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
        "Filter", {output_channels, input_channels, kernel_h, kernel_w});
    net.AddRandomInput<float>("Bias", {output_channels});
    // run cpu
    net.RunOp();

    // Check
    Tensor expected;
    expected.Copy(*net.GetOutput("Output"));

    // Run NEON
    net.RunOp(DeviceType::NEON);
    ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 0.001);
  };

  for (int kernel_size : {1, 3, 5}) {
    for (int stride : {1, 2}) {
      func(kernel_size, kernel_size, stride, stride, VALID);
      func(kernel_size, kernel_size, stride, stride, SAME);
    }
  }
}
