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
  OpsTestNet net;
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
      "Input", {1, 4, 4, 2},
      {0, 16, 1, 17, 2,  18, 3,  19, 4,  20, 5,  21, 6,  22, 7,  23,
       8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31});

  // Run
  net.RunOp();

  // Check
  auto expected =
      CreateTensor<float>({1, 2, 2, 2}, {5, 21, 7, 23, 13, 29, 15, 31});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, MAX_SAME) {
  // Construct graph
  OpsTestNet net;
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
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 3, 3, 1},
                                                {0, 1, 2, 3, 4, 5, 6, 7, 8});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 2, 2, 1}, {4, 5, 7, 8});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, MAX_VALID_DILATION) {
  // Construct graph
  OpsTestNet net;
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
      "Input", {1, 4, 4, 1},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 2, 2, 1}, {10, 11, 14, 15});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, MAX_k2x2s2x2) {
  // Construct graph
  OpsTestNet net;
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
      "Input", {1, 2, 9, 1},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 1, 5, 1}, {10, 12, 14, 16, 17});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

template <DeviceType D>
static void SimpleMaxPooling3S2() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 3, 9, 1},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
       14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT);
    OpDefBuilder("Pooling", "PoolingTest")
        .Input("InputImage")
        .Output("OutputImage")
        .AddIntArg("pooling_type", PoolingType::MAX)
        .AddIntsArg("kernels", {3, 3})
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    net.RunOp(D);
    ImageToBuffer<D, float>(net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT);
  } else {
    // Run
    OpDefBuilder("Pooling", "PoolingTest")
        .Input("Input")
        .Output("Output")
        .AddIntArg("pooling_type", PoolingType::MAX)
        .AddIntsArg("kernels", {3, 3})
        .AddIntsArg("strides", {2, 2})
        .AddIntArg("padding", Padding::VALID)
        .AddIntsArg("dilations", {1, 1})
        .Finalize(net.NewOperatorDef());
    net.RunOp(D);
  }

  // Check
  auto expected = CreateTensor<float>({1, 1, 4, 1}, {20, 22, 24, 26});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, CPUSimpleMaxPooling3S2) { SimpleMaxPooling3S2<CPU>(); }

TEST_F(PoolingOpTest, OPENCLSimpleMaxPooling3S2) {
  SimpleMaxPooling3S2<OPENCL>();
}

template <DeviceType D, typename T>
static void MaxPooling3S2(const std::vector<index_t> &input_shape,
                          const std::vector<int> strides,
                          Padding padding) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::MAX)
      .AddIntsArg("kernels", {3, 3})
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, T>("Input", input_shape);

  // run on cpu
  net.RunOp();
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  BufferToImage<D, T>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntArg("pooling_type", PoolingType::MAX)
      .AddIntsArg("kernels", {3, 3})
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  net.RunOp(D);
  ImageToBuffer<D, T>(net, "OutputImage", "OPENCLOutput",
                      kernels::BufferType::IN_OUT);

  ExpectTensorNear<T>(expected, *net.GetOutput("OPENCLOutput"), 0.001);
}

// TODO(chenghui) : there is a bug.
// TEST_F(PoolingOpTest, NEONAlignedMaxPooling3S2) {
//  AlignedMaxPooling3S2<NEON>(Padding::VALID);
//  AlignedMaxPooling3S2<NEON>(Padding::SAME);
//}

TEST_F(PoolingOpTest, OPENCLAlignedMaxPooling3S2) {
  MaxPooling3S2<OPENCL, float>({3, 64, 32, 32}, {1, 1}, Padding::VALID);
  MaxPooling3S2<OPENCL, float>({3, 64, 32, 32}, {2, 2}, Padding::VALID);
  MaxPooling3S2<OPENCL, float>({3, 64, 32, 32}, {1, 1}, Padding::SAME);
  MaxPooling3S2<OPENCL, float>({3, 64, 32, 32}, {2, 2}, Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLHalfAlignedMaxPooling3S2) {
  MaxPooling3S2<OPENCL, half>({3, 64, 32, 32}, {1, 1}, Padding::VALID);
  MaxPooling3S2<OPENCL, half>({3, 64, 32, 32}, {2, 2}, Padding::VALID);
  MaxPooling3S2<OPENCL, half>({3, 64, 32, 32}, {1, 1}, Padding::SAME);
  MaxPooling3S2<OPENCL, half>({3, 64, 32, 32}, {2, 2}, Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLUnalignedMaxPooling3S2) {
  MaxPooling3S2<OPENCL, half>({3, 41, 43, 47}, {1, 1}, Padding::VALID);
  MaxPooling3S2<OPENCL, half>({3, 41, 43, 47}, {2, 2}, Padding::VALID);
  MaxPooling3S2<OPENCL, half>({3, 41, 43, 47}, {1, 1}, Padding::SAME);
  MaxPooling3S2<OPENCL, half>({3, 41, 43, 47}, {2, 2}, Padding::SAME);
}

TEST_F(PoolingOpTest, AVG_VALID) {
  // Construct graph
  OpsTestNet net;
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
      "Input", {1, 4, 4, 2},
      {0, 16, 1, 17, 2,  18, 3,  19, 4,  20, 5,  21, 6,  22, 7,  23,
       8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>(
      {1, 2, 2, 2}, {2.5, 18.5, 4.5, 20.5, 10.5, 26.5, 12.5, 28.5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

template <DeviceType D>
static void SimpleAvgPoolingTest() {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>(
      "Input", {1, 2, 8, 1},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  BufferToImage<D, float>(net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT);
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntArg("pooling_type", PoolingType::AVG)
      .AddIntsArg("kernels", {2, 2})
      .AddIntsArg("strides", {2, 2})
      .AddIntArg("padding", Padding::SAME)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());
  // Run
  net.RunOp(D);
  ImageToBuffer<D, float>(net, "OutputImage", "Output",
                          kernels::BufferType::IN_OUT);

  // Check
  auto expected = CreateTensor<float>({1, 1, 4, 1}, {4.5, 6.5, 8.5, 10.5});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(PoolingOpTest, OPENCLSimpleAvgPooling) {
  SimpleAvgPoolingTest<OPENCL>();
}

template <DeviceType D, typename T>
static void AvgPoolingTest(const std::vector<index_t> &shape,
                           const std::vector<int> &kernels,
                           const std::vector<int> &strides,
                           Padding padding) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("pooling_type", PoolingType::AVG)
      .AddIntsArg("kernels", kernels)
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<D, float>("Input", shape);

  // run on cpu
  net.RunOp();
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  BufferToImage<D, T>(net, "Input", "InputImage", kernels::BufferType::IN_OUT);
  OpDefBuilder("Pooling", "PoolingTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntArg("pooling_type", PoolingType::AVG)
      .AddIntsArg("kernels", kernels)
      .AddIntsArg("strides", strides)
      .AddIntArg("padding", padding)
      .AddIntsArg("dilations", {1, 1})
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());
  net.RunOp(D);
  ImageToBuffer<D, T>(net, "OutputImage", "OPENCLOutput",
                      kernels::BufferType::IN_OUT);

  ExpectTensorNear<float, T>(expected, *net.GetOutput("OPENCLOutput"), 0.01);
}

TEST_F(PoolingOpTest, OPENCLAlignedAvgPooling) {
  AvgPoolingTest<OPENCL, float>({3, 15, 15, 128}, {4, 4}, {4, 4},
                                Padding::VALID);
  AvgPoolingTest<OPENCL, float>({3, 15, 15, 128}, {4, 4}, {4, 4},
                                Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLHalfAlignedAvgPooling) {
  AvgPoolingTest<OPENCL, half>({3, 15, 15, 128}, {4, 4}, {4, 4},
                               Padding::VALID);
  AvgPoolingTest<OPENCL, half>({3, 15, 15, 128}, {4, 4}, {4, 4}, Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLAlignedLargeKernelAvgPooling) {
  AvgPoolingTest<OPENCL, float>({3, 64, 64, 128}, {16, 16}, {16, 16},
                                Padding::VALID);
  AvgPoolingTest<OPENCL, float>({3, 64, 64, 128}, {16, 16}, {16, 16},
                                Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLHalfAlignedLargeKernelAvgPooling) {
  AvgPoolingTest<OPENCL, half>({3, 64, 64, 128}, {16, 16}, {16, 16},
                               Padding::VALID);
  AvgPoolingTest<OPENCL, half>({3, 64, 64, 128}, {16, 16}, {16, 16},
                               Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLUnAlignedAvgPooling) {
  AvgPoolingTest<OPENCL, float>({3, 31, 37, 128}, {2, 2}, {2, 2},
                                Padding::VALID);
  AvgPoolingTest<OPENCL, float>({3, 31, 37, 128}, {2, 2}, {2, 2},
                                Padding::SAME);
}

TEST_F(PoolingOpTest, OPENCLUnAlignedLargeKernelAvgPooling) {
  AvgPoolingTest<OPENCL, float>({3, 31, 37, 128}, {8, 8}, {8, 8},
                                Padding::VALID);
  AvgPoolingTest<OPENCL, float>({3, 31, 37, 128}, {8, 8}, {8, 8},
                                Padding::SAME);
}
