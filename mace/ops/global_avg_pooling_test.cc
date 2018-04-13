//
// Copyright (c) 2017 XiaoMi All rights reserved.
//
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {
namespace ops {
namespace test {

class GlobalAvgPoolingOpTest : public OpsTestBase {};

TEST_F(GlobalAvgPoolingOpTest, 3x7x7_CPU) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("GlobalAvgPooling", "GlobalAvgPoolingTest")
      .Input("Input")
      .Output("Output")
      .Finalize(net.NewOperatorDef());

  // Add input data
  std::vector<float> input(147);
  for (int i = 0; i < 147; ++i) {
    input[i] = i / 49 + 1;
  }
  net.AddInputFromArray<DeviceType::CPU, float>("Input", {1, 3, 7, 7}, input);

  // Run
  net.RunOp();

  // Check
  auto expected = CreateTensor<float>({1, 3, 1, 1}, {1, 2, 3});

  ExpectTensorNear<float>(*expected, *net.GetOutput("Output"), 1e-5);
}

}  // namespace test
}  // namespace ops
}  // namespace mace
