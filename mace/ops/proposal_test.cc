//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {

class ProposalOpTest : public OpsTestBase {};

void TestSimple() {
  OpsTestNet net;

  OpDefBuilder("Proposal", "ProposalTest")
      .Input("RpnCLSProb")
      .Input("RpnBBoxPred")
      .Input("IMInfo")
      .AddIntArg("feat_stride", 16)
      .AddIntsArg("scales", {2, 4, 8, 16, 32})
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddInputFromArray<DeviceType::CPU, float>(
      "RpnCLSProb", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});
  net.AddInputFromArray<DeviceType::CPU, float>(
      "RpnBBoxPred", {2, 2, 2, 2},
      {-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 0});
  net.AddInputFromArray<DeviceType::CPU, float>(
      "IMInfo", {2, 2},
      {1, 1, 1, 1});

  // Run
  net.RunOp();

  auto expected = CreateTensor<float>(
      {2, 2, 2, 2}, {0, 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0});

}

TEST_F(ProposalOpTest, CPUSimple) { TestSimple(); }


}  // namespace mace
