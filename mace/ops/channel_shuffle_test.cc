//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class ChannelShuffleOpTest : public OpsTestBase {};

TEST_F(ChannelShuffleOpTest, C8G4_CPU) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("ChannelShuffle", "ChannelShuffleTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("group", 4)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 1, 2, 8},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>(
      {1, 1, 2, 8}, {0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}

TEST_F(ChannelShuffleOpTest, C16G4_OPENCL) {
  // Construct graph
  OpsTestNet net;

  // Add input data
  net.AddInputFromArray<DeviceType::OPENCL, float>(
    "Input", {1, 1, 2, 16},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
  BufferToImage<DeviceType::OPENCL, float>(net, "Input", "InputImage",
                          kernels::BufferType::IN_OUT_CHANNEL);


  OpDefBuilder("ChannelShuffle", "ChannelShuffleTest")
    .Input("InputImage")
    .Output("OutputImage")
    .AddIntArg("group", 4)
    .Finalize(net.NewOperatorDef());

  // Run
  net.RunOp(DeviceType::OPENCL);

  // Transfer output
  ImageToBuffer<DeviceType::OPENCL, float>(net, "OutputImage", "Output",
                          kernels::BufferType::IN_OUT_CHANNEL);

  // Check
  auto expected = CreateTensor<float>(
    {1, 1, 2, 16}, {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
    16, 20, 24, 28, 17, 21, 25, 29, 18, 22, 26, 30, 19, 23, 27, 31});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}