//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <fstream>
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {

class GEMMOpTest : public OpsTestBase {};

template<DeviceType D>
void Simple(const std::vector<index_t> &A_shape,
            const std::vector<float> &A_value,
            const std::vector<index_t> &B_shape,
            const std::vector<float> &B_value,
            const std::vector<index_t> &C_shape,
            const std::vector<float> &C_value) {
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<D, float>("A", A_shape, A_value);
  net.AddInputFromArray<D, float>("B", B_shape, B_value);

  if (D == DeviceType::OPENCL) {
    BufferToImage<D, float>(net, "A", "AImage",
                            kernels::BufferType::IN_OUT_WIDTH);
    BufferToImage<D, float>(net, "B", "BImage",
                            kernels::BufferType::IN_OUT_HEIGHT);

    OpDefBuilder("GEMM", "GEMMTest")
        .Input("AImage")
        .Input("BImage")
        .Output("OutputImage")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);

    // Transfer output
    ImageToBuffer<D, float>(net, "OutputImage", "Output",
                            kernels::BufferType::IN_OUT_HEIGHT);
  } else {
    OpDefBuilder("GEMM", "GEMMTest")
        .Input("A")
        .Input("B")
        .Output("Output")
        .Finalize(net.NewOperatorDef());
    // Run
    net.RunOp(D);
  }

  // Check
  auto expected =
      CreateTensor<float>(C_shape, C_value);

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

TEST_F(GEMMOpTest, SimpleCPU) {
  Simple<DeviceType::CPU>({1, 2, 3, 1}, {1, 2, 3, 4, 5, 6},
                          {1, 3, 2, 1}, {1, 2, 3, 4, 5, 6},
                          {1, 2, 2, 1}, {22, 28, 49, 64});
  Simple<DeviceType::CPU>({1, 5, 5, 1},
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                          {1, 5, 5, 1},
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                          {1, 5, 5, 1},
                          {215, 230, 245, 260, 275, 490, 530, 570, 610, 650,
                           765, 830, 895, 960, 1025, 1040, 1130, 1220, 1310, 1400,
                           1315, 1430, 1545, 1660, 1775});
}


TEST_F(GEMMOpTest, SimpleCPUWithBatch) {
  Simple<DeviceType::CPU>({2, 2, 3, 1}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6},
                          {2, 3, 2, 1}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6},
                          {2, 2, 2, 1}, {22, 28, 49, 64, 22, 28, 49, 64});
}

TEST_F(GEMMOpTest, SimpleOPENCL) {
  Simple<DeviceType::OPENCL>({1, 2, 3, 1}, {1, 2, 3, 4, 5, 6},
                             {1, 3, 2, 1}, {1, 2, 3, 4, 5, 6},
                             {1, 2, 2, 1}, {22, 28, 49, 64});
  Simple<DeviceType::OPENCL>({1, 5, 5, 1},
                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                             {1, 5, 5, 1},
                             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                              16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                             {1, 5, 5, 1},
                             {215, 230, 245, 260, 275, 490, 530, 570, 610, 650,
                              765, 830, 895, 960, 1025, 1040, 1130, 1220, 1310, 1400,
                              1315, 1430, 1545, 1660, 1775});
}

TEST_F(GEMMOpTest, SimpleGPUWithBatch) {
  Simple<DeviceType::CPU>({2, 2, 3, 1}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6},
                          {2, 3, 2, 1}, {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6},
                          {2, 2, 2, 1}, {22, 28, 49, 64, 22, 28, 49, 64});
}

template <typename T>
void Complex(const index_t batch,
             const index_t height,
             const index_t channels,
             const index_t out_width) {
  srand(time(NULL));

  // Construct graph
  OpsTestNet net;
  OpDefBuilder("GEMM", "GEMMTest")
      .Input("A")
      .Input("B")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::OPENCL, float>(
      "A", {batch, height, channels, 1});
  net.AddRandomInput<DeviceType::OPENCL, float>(
      "B", {batch, channels, out_width, 1});

  // run cpu
  net.RunOp();

  // Check
  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Run on opencl
  BufferToImage<DeviceType::OPENCL, T>(net, "A", "AImage",
                                           kernels::BufferType::IN_OUT_WIDTH);
  BufferToImage<DeviceType::OPENCL, T>(net, "B", "BImage",
                                           kernels::BufferType::IN_OUT_HEIGHT);

  OpDefBuilder("GEMM", "GEMMTest")
      .Input("AImage")
      .Input("BImage")
      .Output("OutputImage")
      .AddIntArg("T", static_cast<int>(DataTypeToEnum<T>::value))
      .Finalize(net.NewOperatorDef());

  // Run on opencl
  net.RunOp(DeviceType::OPENCL);

  ImageToBuffer<DeviceType::OPENCL, float>(net, "OutputImage", "OPENCLOutput",
                                           kernels::BufferType::IN_OUT_HEIGHT);
  if (DataTypeToEnum<T>::value == DataType::DT_HALF) {
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-1);
  } else {
    ExpectTensorNear<float>(expected, *net.GetOutput("OPENCLOutput"), 1e-4);
  }
}

TEST_F(GEMMOpTest, OPENCLAlignedWithoutBatch) {
  Complex<float>(1, 64, 128, 32);
  Complex<float>(1, 64, 32, 128);
}
TEST_F(GEMMOpTest, OPENCLUnAlignedWithoutBatch) {
  Complex<float>(1, 31, 113, 61);
  Complex<float>(1, 113, 31, 73);
}
TEST_F(GEMMOpTest, OPENCLUnAlignedWithBatch) {
  Complex<float>(2, 3, 3, 3);
  Complex<float>(16, 31, 61, 67);
  Complex<float>(31, 31, 61, 67);
}
TEST_F(GEMMOpTest, OPENCLHalfAlignedWithoutBatch) {
  Complex<half>(1, 64, 128, 32);
  Complex<half>(1, 64, 32, 128);
}
TEST_F(GEMMOpTest, OPENCLHalfUnAlignedWithBatch) {
  Complex<half>(2, 31, 113, 61);
  Complex<half>(16, 32, 64, 64);
  Complex<half>(31, 31, 61, 67);
}

}
