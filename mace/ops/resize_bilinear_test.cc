//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/resize_bilinear.h"
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class ResizeBilinearTest : public OpsTestBase {};

TEST_F(ResizeBilinearTest, ResizeBilinearWOAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("Input")
      .Input("OutSize")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<float>("Input", {1, 3, 2, 4}, input);
  net.AddInputFromArray<index_t>("OutSize", {2}, {1, 2});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 3, 1, 2}, {0, 2, 8, 10, 16, 18});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(ResizeBilinearTest, ResizeBilinearWAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("Input")
      .Input("OutSize")
      .Output("Output")
      .Finalize(net.operator_def());

  net.AddIntArg("align_corners", 1);

  // Add input data
  vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  net.AddInputFromArray<float>("Input", {1, 3, 2, 4}, input);
  net.AddInputFromArray<index_t>("OutSize", {2}, {1, 2});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 3, 1, 2}, {0, 3, 8, 11, 16, 19});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}
