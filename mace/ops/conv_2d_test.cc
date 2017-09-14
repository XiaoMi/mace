//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/conv_2d.h"
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
  net.AddInputFromArray<float>("Input", {1, 2, 3, 3},
                    {1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1});
  net.AddInputFromArray<float>("Filter", {1, 2, 3, 3},
                           {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<float>("Bias", {1}, {0.1f});

  // Run
  net.RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 1, 1, 1}, {18.1f});

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 0.001);
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
  net.AddInputFromArray<float>("Input", {1, 2, 3, 3},
                    {1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1});
  net.AddInputFromArray<float>("Filter", {1, 2, 3, 3},
                           {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  net.AddInputFromArray<float>("Bias", {1}, {0.1f});

  // Run
  net.RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 1, 3, 3},
                                        { 8.1f, 12.1f,  8.1f,
                                         12.1f, 18.1f, 12.1f,
                                          8.1f, 12.1f,  8.1f});

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 0.001);
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
  net.AddInputFromArray<float>("Input", {1, 2, 5, 5},
                    {1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1});
  net.AddInputFromArray<float>("Filter", {2, 2, 3, 3},
                           {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                            0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
  net.AddInputFromArray<float>("Bias", {2}, {0.1f, 0.2f});

  // Run
  net.RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 2, 3, 3},
                                        { 8.1f, 12.1f,  8.1f,
                                         12.1f, 18.1f, 12.1f,
                                          8.1f, 12.1f,  8.1f,
                                          4.2f, 6.2f, 4.2f,
                                          6.2f, 9.2f, 6.2f,
                                          4.2f, 6.2f, 4.2f});


  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 0.001);
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
  net.AddInputFromArray<float>("Input", {1, 5, 3, 10},
                    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  net.AddInputFromArray<float>("Filter", {2, 5, 1, 1},
                           {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            2.0f, 2.0f, 2.0f, 2.0f, 2.0f});
  net.AddInputFromArray<float>("Bias", {2}, {0.1f, 0.2f});

  // Run
  net.RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 2, 3, 10},
                                        {5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f,
                                         5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f,
                                         5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f,
                                         10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f,
                                         10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f,
                                         10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f});

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 0.001);
}

// TODO we need more tests
TEST_F(Conv2dOpTest, Conv3x3R1) {
  auto func = [&](Padding type) {
    srand(time(NULL));

    // generate random input
    index_t batch = 1 + rand() % 5;
    index_t input_channels = 3 + rand() % 50;
    index_t height = 10 + rand() % 100;
    index_t width = 10 + rand() % 100;
    index_t output_channels = 3 + rand() % 50;
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
    net.AddIntArg("padding", type);
    net.AddIntsArg("dilations", {1, 1});

    // Add input data
    net.AddRandomInput<float>("Input", {batch, input_channels, height, width});
    net.AddRandomInput<float>("Filter", {output_channels, input_channels, 3, 3});
    net.AddRandomInput<float>("Bias", {output_channels});
    // run cpu
    net.RunOp();

    // Check
    Tensor expected = *net.GetOutput("Output");

    // Run NEON
    net.RunOp(DeviceType::NEON);

    ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 1e-5);

  };

  func(VALID);
  func(SAME);
}
