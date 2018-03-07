//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "gmock/gmock.h"
#include "mace/core/operator.h"
#include "mace/ops/ops_test_util.h"

using namespace mace;

class ReshapeTest : public OpsTestBase {};

void TestReshape(const std::vector<index_t> &org_shape,
                 const std::vector<int> &output_shape,
                 const std::vector<index_t> &res_shape) {
  // Construct graph
  OpsTestNet net;
  OpDefBuilder("Reshape", "ReshapeTest")
      .Input("Input")
      .Output("Output")
      .AddIntsArg("shape", output_shape)
      .Finalize(net.NewOperatorDef());

  // Add input data
  net.AddRandomInput<DeviceType::CPU, float>("Input", org_shape);

  // Run
  net.RunOp();

  auto input = net.GetTensor("Input");
  auto output = net.GetTensor("Output");

  EXPECT_THAT(output->shape(), ::testing::ContainerEq(res_shape));

  const float *input_ptr = input->data<float>();
  const float *output_ptr = output->data<float>();
  const int size = output->size();
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(input_ptr[i], output_ptr[i]);
  }
}

TEST_F(ReshapeTest, Simple) {
  TestReshape({1, 2, 3, 4}, {1, 2, -1, 4}, {1, 2, 3, 4});
  TestReshape({1, 2, 3, 4}, {1, 2, -1, 2}, {1, 2, 6, 2});
  TestReshape({1, 2, 3, 4}, {1, -1, 3, 2}, {1, 4, 3, 2});
  TestReshape({1, 2, 3, 4}, {2, 2, 3, 2}, {2, 2, 3, 2});
}

TEST_F(ReshapeTest, Complex) {
  TestReshape({1, 2, 3, 4}, {-1}, {24});
  TestReshape({1, 2, 3, 4}, {1, -1}, {1, 24});
  TestReshape({1, 2, 3, 4}, {-1, 1}, {24, 1});
  TestReshape({1, 2, 3, 4}, {1, 3, 8}, {1, 3, 8});
}
