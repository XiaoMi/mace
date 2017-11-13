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
  auto &net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("pooling_type", PoolingType::MAX)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
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
  auto &net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("pooling_type", PoolingType::AVG)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
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
  auto &net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("pooling_type", PoolingType::MAX)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 1, 3, 3},
                               {0, 1, 2, 3, 4, 5, 6, 7, 8});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 1, 2, 2}, {4, 5, 7, 8});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, MAX_VALID_DILATION) {
  // Construct graph
  auto &net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {1, 1})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {2, 2})
      .AddIntArg("pooling_type", PoolingType::MAX)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
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
  auto &net = test_net();
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::MAX)
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 1, 2, 9},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  // Run
  net.RunOp(DeviceType::NEON);

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 5}, {10, 12, 14, 16, 17});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}


template <DeviceType D>
static void SimpleMaxPooling3S2() {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::MAX)
      .AddIntsArg("kernels", {3, 3})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::VALID)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 1, 3, 9},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
       14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});
  // Run
  net.RunOp(D);

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 4}, {20, 22, 24, 26});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, CPUSimpleMaxPooling3S2) {
  SimpleMaxPooling3S2<CPU>();
}
TEST_F(PoolingOpTest, NEONSimpleMaxPooling3S2) {
  SimpleMaxPooling3S2<NEON>();
}
TEST_F(PoolingOpTest, OPENCLSimpleMaxPooling3S2) {
  SimpleMaxPooling3S2<OPENCL>();
}

template <DeviceType D>
static void AlignedMaxPooling3S2(Padding padding) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::MAX)
      .AddIntsArg("kernels", {3, 3})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, float>("Input", {3, 128, 64, 64});
  // Run
  net.RunOp(D);
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on cpu
  net.RunOp();

  ExpectTensorNear<float>(*net.GetOutput("Output"), expected, 0.001);
}

// TODO(chenghui) : there is a bug.
//TEST_F(PoolingOpTest, NEONAlignedMaxPooling3S2) {
//  AlignedMaxPooling3S2<NEON>(Padding::VALID);
//  AlignedMaxPooling3S2<NEON>(Padding::SAME);
//}

TEST_F(PoolingOpTest, OPENCLAlignedMaxPooling3S2) {
  AlignedMaxPooling3S2<OPENCL>(Padding::VALID);
  AlignedMaxPooling3S2<OPENCL>(Padding::SAME);
}

template <DeviceType D>
static void UnalignedMaxPooling3S2(Padding padding) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::MAX)
      .AddIntsArg("kernels", {3, 3})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, float>("Input", {3, 113, 43, 47});
  // Run
  net.RunOp(D);
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on cpu
  net.RunOp();

  ExpectTensorNear<float>(*net.GetOutput("Output"), expected, 0.001);
}

// TODO(chenghui) : there is a bug.
//TEST_F(PoolingOpTest, NEONUnalignedMaxPooling3S2) {
//  UnalignedMaxPooling3S2<NEON>();
//}

TEST_F(PoolingOpTest, OPENCLUnalignedMaxPooling3S2) {
  UnalignedMaxPooling3S2<OPENCL>(Padding::VALID);
  UnalignedMaxPooling3S2<OPENCL>(Padding::SAME);
}

template <DeviceType D>
static void SimpleAvgPoolingTest() {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::AVG)
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 1, 2, 8},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  // Run
  net.RunOp(D);

  // Check
  auto expected = CreateTensor<float>({1, 1, 1, 4}, {4.5, 6.5, 8.5, 10.5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, NEONSimpleAvgPooling) {
  SimpleAvgPoolingTest<NEON>();
}

TEST_F(PoolingOpTest, OPENCLSimpleAvgPooling) {
  SimpleAvgPoolingTest<OPENCL>();
}

template <DeviceType D>
static void AlignedAvgPoolingTest(Padding padding) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::AVG)
      .AddIntsArg("kernels", {4, 4})
      .AddIntsArg("strides", {4, 4})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, float>("Input", {3, 128, 15, 15});
  // Run
  net.RunOp(D);
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on cpu
  net.RunOp();

  ExpectTensorNear<float>(*net.GetOutput("Output"), expected, 1e-5);
}

TEST_F(PoolingOpTest, NEONAlignedAvgPooling) {
  AlignedAvgPoolingTest<NEON>(Padding::VALID);
  AlignedAvgPoolingTest<NEON>(Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLAlignedAvgPooling) {
  AlignedAvgPoolingTest<OPENCL>(Padding::VALID);
  AlignedAvgPoolingTest<OPENCL>(Padding::SAME);
}

template <DeviceType D>
static void UnAlignedAvgPoolingTest(Padding padding) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::AVG)
      .AddIntsArg("kernels", {7, 7})
      .AddIntsArg("strides", {7, 7})
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, float>("Input", {3, 128, 31, 37});
  // Run
  net.RunOp(D);
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on cpu
  net.RunOp();

  ExpectTensorNear<float>(*net.GetOutput("Output"), expected, 1e-5);
}

TEST_F(PoolingOpTest, NEONUnAlignedAvgPooling) {
  UnAlignedAvgPoolingTest<NEON>(Padding::VALID);
  UnAlignedAvgPoolingTest<NEON>(Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLUnAlignedAvgPooling) {
  UnAlignedAvgPoolingTest<OPENCL>(Padding::VALID);
  UnAlignedAvgPoolingTest<OPENCL>(Padding::SAME);
}
