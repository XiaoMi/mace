//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"
#include "mace/ops/conv_2d.h"

using namespace mace;

class Conv2dOpTest : public OpsTestBase {};

TEST_F(Conv2dOpTest, Simple_VALID) {
  // Construct graph
  OpDefBuilder("Conv2d", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .Finalize(operator_def());

  // Add args
  AddIntsArg("strides", {1, 1});
  AddIntArg("padding", Padding::VALID);
  AddIntsArg("dilations", {1, 1});

  // Add input data
  AddInputFromArray<float>("Input", {1, 2, 3, 3},
                    {1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1});
  AddInputFromArray<float>("Filter", {1, 2, 3, 3},
                           {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  AddInputFromArray<float>("Bias", {1}, {0.1f});

  // Run
  RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 1, 1, 1}, {18.1f});

  ExpectTensorNear<float>(expected, *GetOutput("Output"), 0.001);
}

TEST_F(Conv2dOpTest, Simple_SAME) {
  // Construct graph
  OpDefBuilder("Conv2d", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .Finalize(operator_def());

  // Add args
  AddIntsArg("strides", {1, 1});
  AddIntArg("padding", Padding::SAME);
  AddIntsArg("dilations", {1, 1});

  // Add input data
  AddInputFromArray<float>("Input", {1, 2, 3, 3},
                    {1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1,
                     1, 1, 1});
  AddInputFromArray<float>("Filter", {1, 2, 3, 3},
                           {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  AddInputFromArray<float>("Bias", {1}, {0.1f});

  // Run
  RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 1, 3, 3},
                                        { 8.1f, 12.1f,  8.1f,
                                         12.1f, 18.1f, 12.1f,
                                          8.1f, 12.1f,  8.1f});

  ExpectTensorNear<float>(expected, *GetOutput("Output"), 0.001);
}

TEST_F(Conv2dOpTest, Combined) {
  // Construct graph
  OpDefBuilder("Conv2d", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .Finalize(operator_def());

  // Add args
  AddIntsArg("strides", {2, 2});
  AddIntArg("padding", Padding::SAME);
  AddIntsArg("dilations", {1, 1});

  // Add input data
  AddInputFromArray<float>("Input", {1, 2, 5, 5},
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
  AddInputFromArray<float>("Filter", {2, 2, 3, 3},
                           {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                            0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
  AddInputFromArray<float>("Bias", {2}, {0.1f, 0.2f});

  // Run
  RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 2, 3, 3},
                                        { 8.1f, 12.1f,  8.1f,
                                         12.1f, 18.1f, 12.1f,
                                          8.1f, 12.1f,  8.1f,
                                          4.2f, 6.2f, 4.2f,
                                          6.2f, 9.2f, 6.2f,
                                          4.2f, 6.2f, 4.2f});


  ExpectTensorNear<float>(expected, *GetOutput("Output"), 0.001);
}

TEST_F(Conv2dOpTest, Conv1x1) {
  // Construct graph
  OpDefBuilder("Conv2d", "Conv2dTest")
        .Input("Input")
        .Input("Filter")
        .Input("Bias")
        .Output("Output")
        .Finalize(operator_def());

  // Add args
  AddIntsArg("strides", {1, 1});
  AddIntArg("padding", Padding::VALID);
  AddIntsArg("dilations", {1, 1});

  // Add input data
  AddInputFromArray<float>("Input", {1, 5, 3, 10},
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
  AddInputFromArray<float>("Filter", {2, 5, 1, 1},
                           {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                            2.0f, 2.0f, 2.0f, 2.0f, 2.0f});
  AddInputFromArray<float>("Bias", {2}, {0.1f, 0.2f});

  // Run
  RunOp(DeviceType::NEON);

  // Check
  Tensor expected = CreateTensor<float>({1, 2, 3, 10},
                                        {5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f,
                                         5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f,
                                         5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f, 5.1f,
                                         10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f,
                                         10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f,
                                         10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f, 10.2f});

  ExpectTensorNear<float>(expected, *GetOutput("Output"), 0.001);
}

// TODO we need more tests
