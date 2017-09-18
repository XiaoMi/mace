//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "gtest/gtest.h"

#include "mace/core/operator.h"
#include "mace/kernels/pooling.h"
#include "mace/ops/conv_pool_2d_base.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class PoolingOpTest : public OpsTestBase {};

TEST_F(PoolingOpTest, MAX_VALID) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add args
  net.AddIntsArg("kernels", {2, 2});
  net.AddIntsArg("strides", {2, 2});
  net.AddIntArg("padding", Padding::VALID);
  net.AddIntsArg("dilations", {1, 1});
  net.AddIntArg("pooling_type", PoolingType::MAX);

  // Add input data
  net.AddInputFromArray<float>(
      "Input", {1, 2, 4, 4},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});

  // Run
  net.RunOp();

  // Check
  auto expected =
      CreateTensor<float>({1, 2, 2, 2}, {5, 7, 13, 15, 21, 23, 29, 31});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, AVG_VALID) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add args
  net.AddIntsArg("kernels", {2, 2});
  net.AddIntsArg("strides", {2, 2});
  net.AddIntArg("padding", Padding::VALID);
  net.AddIntsArg("dilations", {1, 1});
  net.AddIntArg("pooling_type", PoolingType::AVG);

  // Add input data
  net.AddInputFromArray<float>(
      "Input", {1, 2, 4, 4},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>(
      {1, 2, 2, 2}, {2.5, 4.5, 10.5, 12.5, 18.5, 20.5, 26.5, 28.5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, MAX_SAME) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add args
  net.AddIntsArg("kernels", {2, 2});
  net.AddIntsArg("strides", {2, 2});
  net.AddIntArg("padding", Padding::SAME);
  net.AddIntsArg("dilations", {1, 1});
  net.AddIntArg("pooling_type", PoolingType::MAX);

  // Add input data
  net.AddInputFromArray<float>("Input", {1, 1, 3, 3},
                               {0, 1, 2, 3, 4, 5, 6, 7, 8});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 1, 2, 2}, {4, 5, 7, 8});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, MAX_VALID_DILATION) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add args
  net.AddIntsArg("kernels", {2, 2});
  net.AddIntsArg("strides", {1, 1});
  net.AddIntArg("padding", Padding::VALID);
  net.AddIntsArg("dilations", {2, 2});
  net.AddIntArg("pooling_type", PoolingType::MAX);

  // Add input data
  net.AddInputFromArray<float>(
      "Input", {1, 1, 4, 4},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 1, 2, 2}, {10, 11, 14, 15});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, MAX_k2x2s2x2) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add args
  net.AddIntArg("pooling_type", PoolingType::MAX);
  net.AddIntsArg("kernels", {2, 2});
  net.AddIntsArg("strides", {2, 2});
  net.AddIntArg("padding", Padding::SAME);
  net.AddIntsArg("dilations", {1, 1});

  // Add input data
  net.AddInputFromArray<float>("Input", {1, 1, 2, 9},
                               {0, 1, 2, 3, 4, 5, 6, 7, 8,
                                9, 10, 11, 12, 13, 14, 15, 16, 17});
  // Run
  net.RunOp(DeviceType::NEON);

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 5}, {10, 12, 14, 16, 17});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, MAX_k3x3s2x2) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add args
  net.AddIntArg("pooling_type", PoolingType::MAX);
  net.AddIntsArg("kernels", {3, 3});
  net.AddIntsArg("strides", {2, 2});
  net.AddIntArg("padding", Padding::VALID);
  net.AddIntsArg("dilations", {1, 1});

  // Add input data
  net.AddInputFromArray<float>("Input", {1, 1, 3, 9},
                               {0, 1, 2, 3, 4, 5, 6, 7, 8,
                                9, 10, 11, 12, 13, 14, 15, 16, 17,
                                18, 19, 20, 21, 22, 23, 24, 25, 26});
  // Run
  net.RunOp(DeviceType::NEON);

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 4}, {20, 22, 24, 26});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, AVG_k2x2s2x2) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add args
  net.AddIntArg("pooling_type", PoolingType::AVG);
  net.AddIntsArg("kernels", {2, 2});
  net.AddIntsArg("strides", {2, 2});
  net.AddIntArg("padding", Padding::SAME);
  net.AddIntsArg("dilations", {1, 1});

  // Add input data
  net.AddInputFromArray<float>(
      "Input", {1, 1, 2, 8},
      {0, 1, 2, 3, 4, 5, 6, 7,
       8, 9, 10, 11, 12, 13, 14, 15});
  // Run
  net.RunOp(DeviceType::NEON);

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 4},
                                      {4.5, 6.5, 8.5, 10.5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

