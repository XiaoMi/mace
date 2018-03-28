//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

template <DeviceType D>
void RunDepthToSpace(const bool d2s,
                     const std::vector<index_t> &input_shape,
                     const std::vector<float> &input_data,
                     const int block_size,
                     const std::vector<index_t> &expected_shape,
                     const std::vector<float> &expected_data) {
  OpsTestNet net;
  net.AddInputFromArray<D, float>("Input", input_shape, input_data);
  const char *ops_name = (d2s) ? "DepthToSpace" : "SpaceToDepth";
  const char *ops_test_name = (d2s) ? "DepthToSpaceTest" : "SpaceToDepthTest";
  // Construct graph
  if (D == DeviceType::CPU) {
    OpDefBuilder(ops_name, ops_test_name)
        .Input("Input")
        .Output("Output")
        .AddIntArg("block_size", block_size)
        .Finalize(net.NewOperatorDef());

  } else {
    BufferToImage<D, float>(&net, "Input", "InputImage",
                            kernels::BufferType::IN_OUT_CHANNEL);
    OpDefBuilder(ops_name, ops_test_name)
        .Input("InputImage")
        .Output("OutputImage")
        .AddIntArg("block_size", block_size)
        .Finalize(net.NewOperatorDef());
  }
  // Run
  net.RunOp(D);

  if (D == DeviceType::OPENCL) {
    ImageToBuffer<DeviceType::OPENCL, float>(&net, "OutputImage", "Output",
        kernels::BufferType::IN_OUT_CHANNEL);
  }
  auto expected = CreateTensor<float>(expected_shape, expected_data);
  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

class SpaceToDepthOpTest : public OpsTestBase {};

TEST_F(SpaceToDepthOpTest, Input2x4x4_B2_CPU) {
  RunDepthToSpace<DeviceType::CPU>(false, {1, 2, 4, 4},
      {0, 1, 2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
       8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31},
      2,
      {1, 1, 2, 16},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
}

TEST_F(SpaceToDepthOpTest, Input2x4x4_B2_OPENCL) {
  RunDepthToSpace<DeviceType::OPENCL>(false, {1, 2, 4, 4},
      {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23,
       8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31},
      2,
      {1, 1, 2, 16},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
}

TEST_F(SpaceToDepthOpTest, Input2x2x4_B2_CPU) {
  RunDepthToSpace<DeviceType::CPU>(false, {1, 2, 2, 4},
      {1, 2, 3, 4, 5, 6, 7, 8,
       9, 10, 11, 12, 13, 14, 15, 16},
      2,
      {1, 1, 1, 16},
      {1, 2, 3, 4, 5, 6, 7, 8,
       9, 10, 11, 12, 13, 14, 15, 16});
}

TEST_F(SpaceToDepthOpTest, Input4x4x1_B2_OPENCL) {
  RunDepthToSpace<DeviceType::OPENCL>(false, {1, 2, 2, 4},
      {1, 2, 3, 4, 5, 6, 7, 8,
       9, 10, 11, 12, 13, 14, 15, 16},
      2,
      {1, 1, 1, 16},
      {1, 2, 3, 4, 5, 6, 7, 8,
       9, 10, 11, 12, 13, 14, 15, 16});
}

class DepthToSpaceOpTest : public OpsTestBase {};

TEST_F(DepthToSpaceOpTest, Input1x2x16_B2_CPU) {
  RunDepthToSpace<DeviceType::CPU>(true, {1, 1, 2, 16},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
      2,
      {1, 2, 4, 4},
      {0, 1, 2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
      8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31});
}

TEST_F(DepthToSpaceOpTest, Input1x2x16_B2_OPENCL) {
  RunDepthToSpace<DeviceType::OPENCL>(true, {1, 1, 2, 16},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
      2,
      {1, 2, 4, 4},
      {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23,
      8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31});
}

TEST_F(DepthToSpaceOpTest, Input1x1x16_B2_CPU) {
  RunDepthToSpace<DeviceType::CPU>(true, {1, 1, 1, 16},
      {1,  2,  3,  4,  5,  6,  7,  8,
       9,  10, 11, 12, 13, 14, 15, 16},
      2,
      {1, 2, 2, 4},
      {1, 2, 3, 4, 5, 6, 7, 8,
       9, 10, 11, 12, 13, 14, 15, 16});
}

TEST_F(DepthToSpaceOpTest, Input1x1x16_B2_OPENCL) {
  RunDepthToSpace<DeviceType::OPENCL>(true, {1, 1, 1, 16},
      {1,  2,  3,  4,  5,  6,  7,  8,
       9,  10, 11, 12, 13, 14, 15, 16},
      2,
      {1, 2, 2, 4},
      {1, 2, 3, 4, 5, 6, 7, 8,
       9, 10, 11, 12, 13, 14, 15, 16});
}

/*
TEST_F(DepthToSpaceOpTest, Input2x2x3_B2_CPU) {

  RunDepthToSpace<DeviceType::CPU>({1, 2, 2, 3},
                                   {1, 2, 3, 4, 5, 6,
									7, 8, 9, 10, 11, 12},
                                   2,
                                   {1, 1, 1, 12},
                                   {1,  2,  3,  4,  5,  6,  7,  8,  
									9,  10, 11, 12});
}

TEST_F(DepthToSpaceOpTest, Input2x2x3_B2_OPENCL) {
  RunDepthToSpace<DeviceType::OPENCL>({1, 2, 2, 6},
                                   {1, 2, 3, 4, 5, 6,
									7, 8, 9, 10, 11, 12
									},
                                   2,
                                   {1, 1, 1, 12},
                                   {1,  2,  3,  4,  5,  6,  7,  8,  
									9,  10, 11, 12});
}

TEST_F(DepthToSpaceOpTest, Input2x2x2_B2_CPU) {

  RunDepthToSpace<DeviceType::CPU>({1, 2, 2, 2},
                                   {1, 10,  2,  20,  3,  30,  4, 40},
                                   2,
                                   {1, 1, 1, 8},
                                   {1, 10,  2,  20,  3,  30,  4, 40});
}

TEST_F(DepthToSpaceOpTest, Input2x2x2_B2_OPENCL) {

  RunDepthToSpace<DeviceType::OPENCL>({1, 2, 2, 2},
                                      {1, 10,  2,  20,  3,  30,  4, 40},
                                      2,
                                      {1, 1, 1, 8},
                                      {1, 10,  2,  20,  3,  30,  4, 40});
}*/
}  // namespace test
}  // namespace ops
}  // namespace mace
