//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class SpaceToDepthOpTest : public OpsTestBase {};

TEST_F(SpaceToDepthOpTest, C8G4_CPU) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("SpaceToDepth", "SpaceToDepthTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("block_size", 2)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 2, 4, 4},
       {0,  1,  2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
	   8,  9,  10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>(
      {1, 1, 2, 16},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(SpaceToDepthOpTest, C16G4_OPENCL) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::OPENCL, float>(
      "Input", {1, 2, 4, 4},
       {0,  1,  2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
	   8,  9,  10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31});
  BufferToImage<DeviceType::OPENCL, float>(&net, "Input", "InputImage",
                                           kernels::BufferType::IN_OUT_CHANNEL);

  OpDefBuilder("SpaceToDepth", "SpaceToDepthTest")
      .Input("InputImage")
      .Output("OutputImage")
      .AddIntArg("block_size", 2)
      .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::OPENCL);

  // Transfer output
  ImageToBuffer<DeviceType::OPENCL, float>(&net, "OutputImage", "Output",
                                           kernels::BufferType::IN_OUT_CHANNEL);

  // Check
  auto expected = CreateTensor<float>(
      {1, 1, 2, 16},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
