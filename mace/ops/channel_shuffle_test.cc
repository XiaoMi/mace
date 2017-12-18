//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class ChannelShuffleOpTest : public OpsTestBase {};

TEST_F(ChannelShuffleOpTest, C8G4) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("ChannelShuffle", "ChannelShuffleTest")
      .Input("Input")
      .Output("Output")
      .AddIntArg("group", 4)
      .Finalize(net.NewOperatorDef());


  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "Input", {1, 8, 1, 2},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>(
      {1, 8, 1, 2}, {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 0.001);
}
