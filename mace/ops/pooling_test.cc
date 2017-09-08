//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "gtest/gtest.h"

#include "mace/core/operator.h"
#include "mace/core/net.h"
#include "mace/ops/ops_test_util.h"
#include "mace/ops/conv_pool_2d_base.h"
#include "mace/kernels/pooling.h"

using namespace mace;

class PoolingOpTest : public OpsTestBase {};

TEST_F(PoolingOpTest, MAX_VALID) {
  // Construct graph
  OpDefBuilder("Pooling", "PoolingTest")
        .Input("Input")
        .Output("Output")
        .Finalize(operator_def());

  // Add args
  AddIntsArg("kernels", {2, 2});
  AddIntsArg("strides", {2, 2});
  AddIntArg("padding", Padding::VALID);
  AddIntsArg("dilations", {1, 1});
  AddIntArg("pooling_type", PoolingType::MAX);

  // Add input data
  AddInputFromArray<float>("Input", {1, 2, 4, 4},
                          {0, 1, 2, 3,
                           4, 5, 6, 7,
                           8, 9, 10, 11,
                           12, 13, 14, 15,
                           16, 17, 18, 19,
                           20, 21, 22, 23,
                           24, 25, 26, 27,
                           28, 29, 30, 31});

  // Run
  RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 2, 2, 2}, 
                                        {5, 7, 13, 15, 21, 23, 29, 31});

  ExpectTensorNear<float>(expected, *GetOutput("Output"), 0.001);
}


TEST_F(PoolingOpTest, AVG_VALID) {
  // Construct graph
  OpDefBuilder("Pooling", "PoolingTest")
        .Input("Input")
        .Output("Output")
        .Finalize(operator_def());

  // Add args
  AddIntsArg("kernels", {2, 2});
  AddIntsArg("strides", {2, 2});
  AddIntArg("padding", Padding::VALID);
  AddIntsArg("dilations", {1, 1});
  AddIntArg("pooling_type", PoolingType::AVG);

  // Add input data
  AddInputFromArray<float>("Input", {1, 2, 4, 4},
                          {0, 1, 2, 3,
                           4, 5, 6, 7,
                           8, 9, 10, 11,
                           12, 13, 14, 15,
                           16, 17, 18, 19,
                           20, 21, 22, 23,
                           24, 25, 26, 27,
                           28, 29, 30, 31});

  // Run
  RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 2, 2, 2}, 
                                        {2.5, 4.5, 10.5, 12.5, 18.5, 20.5, 26.5, 28.5});

  ExpectTensorNear<float>(expected, *GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, MAX_SAME) {
  // Construct graph
  OpDefBuilder("Pooling", "PoolingTest")
        .Input("Input")
        .Output("Output")
        .Finalize(operator_def());

  // Add args
  AddIntsArg("kernels", {2, 2});
  AddIntsArg("strides", {2, 2});
  AddIntArg("padding", Padding::SAME);
  AddIntsArg("dilations", {1, 1});
  AddIntArg("pooling_type", PoolingType::MAX);

  // Add input data
  AddInputFromArray<float>("Input", {1, 1, 3, 3},
                          {0, 1, 2, 
                           3, 4, 5, 
                           6, 7, 8});

  // Run
  RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 1, 2, 2}, 
                                        {4, 5, 7, 8});

  ExpectTensorNear<float>(expected, *GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, MAX_VALID_DILATION) {
  // Construct graph
  OpDefBuilder("Pooling", "PoolingTest")
        .Input("Input")
        .Output("Output")
        .Finalize(operator_def());

  // Add args
  AddIntsArg("kernels", {2, 2});
  AddIntsArg("strides", {1, 1});
  AddIntArg("padding", Padding::VALID);
  AddIntsArg("dilations", {2, 2});
  AddIntArg("pooling_type", PoolingType::MAX);

  // Add input data
  AddInputFromArray<float>("Input", {1, 1, 4, 4},
                          {0, 1, 2, 3,
                           4, 5, 6, 7,
                           8, 9, 10, 11,
                           12, 13, 14, 15});

  // Run
  RunOp();

  // Check
  Tensor expected = CreateTensor<float>({1, 1, 2, 2}, 
                                        {10, 11, 14, 15});

  ExpectTensorNear<float>(expected, *GetOutput("Output"), 0.001);
}
