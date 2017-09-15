//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

namespace mace {

class AddnOpTest : public OpsTestBase {};

TEST_F(AddnOpTest, AddnOp) {
  // Construct graph
  auto& net = test_net();
  OpDefBuilder("AddN", "AddNTest")
      .Input("Input1")
      .Input("Input2")
      .Input("Input3")
      .Output("Output")
      .Finalize(net.operator_def());

  // Add input data
  net.AddRandomInput<float>("Input1", {1, 2, 3, 4});
  net.AddRandomInput<float>("Input2", {1, 2, 3, 4});
  net.AddRandomInput<float>("Input3", {1, 2, 3, 4});

  // Run
  net.RunOp();

  Tensor expected;
  expected.Copy(*net.GetOutput("Output"));

  // Check
  net.RunOp(DeviceType::NEON);

  ExpectTensorNear<float>(expected, *net.GetOutput("Output"), 0.01);
}

} // namespace mace
