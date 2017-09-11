//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"
#include "mace/ops/resize_bilinear.h"

using namespace mace;

class ResizeBilinearTest : public OpsTestBase {};

TEST_F(ResizeBilinearTest, ResizeBilinearWOAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("Input")
      .Input("OutSize")
      .Output("Output")
      .Finalize(operator_def());

  // Add input data
  vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  AddInputFromArray<float>("Input", {1, 3, 2, 4}, input);
  AddInputFromArray<index_t>("OutSize", {2}, {1, 2});

  // Run
  RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 3, 1, 2}, {0, 2, 8, 10, 16, 18});

  ExpectTensorNear<float>(expected, *GetOutput("Output"), 0.001);
}

TEST_F(ResizeBilinearTest, ResizeBilinearWAlignCorners) {
  testing::internal::LogToStderr();
  // Construct graph
  OpDefBuilder("ResizeBilinear", "ResizeBilinearTest")
      .Input("Input")
      .Input("OutSize")
      .Output("Output")
      .Finalize(operator_def());

  AddIntArg("align_corners", 1);

  // Add input data
  vector<float> input(24);
  std::iota(begin(input), end(input), 0);
  AddInputFromArray<float>("Input", {1, 3, 2, 4}, input);
  AddInputFromArray<index_t>("OutSize", {2}, {1, 2});

  // Run
  RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 3, 1, 2}, {0, 3, 8, 11, 16, 19});

  ExpectTensorNear<float>(expected, *GetOutput("Output"), 0.001);
}
