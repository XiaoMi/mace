//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class GlobalAvgPoolingOpTest : public OpsTestBase {};

TEST_F(GlobalAvgPoolingOpTest, 3x7x7_CPU) {
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("GlobalAvgPooling", "GlobalAvgPoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  std::vector<float> input(147);
  for (int i = 0; i < 147; ++i) {
    input[i] = i / 49 + 1;
  }
  net.AddInputFromArray<float>("Input", {1, 3, 7, 7}, input);

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 3, 1, 1}, {1, 2, 3});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(GlobalAvgPoolingOpTest, 3x7x7_NEON) {
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("GlobalAvgPooling", "GlobalAvgPoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  std::vector<float> input(147);
  for (int i = 0; i < 147; ++i) {
    input[i] = i / 49 + 1;
  }
  net.AddInputFromArray<float>("Input", {1, 3, 7, 7}, input);

  // Run
  net.RunOp(DeviceType::NEON);

  // Check
  auto expected = CreateTensor<float>({1, 3, 1, 1}, {1, 2, 3});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}
